import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import *
from models import OriginalDQN, ActorCV, CriticCV
from replay_memory import ReplayMemory

def _process_state_helper(state_list_complex_single, target_state_size):
    state_features = []
    for array_np in state_list_complex_single:
        real_part = torch.tensor(array_np.real.flatten(), dtype=torch.float32)
        imag_part = torch.tensor(array_np.imag.flatten(), dtype=torch.float32)
        features = torch.cat([real_part, imag_part], dim=-1)
        state_features.append(features)

    state_vector_flat = torch.cat(state_features, dim=-1)

    current_dim = state_vector_flat.shape[0]
    if current_dim < target_state_size:
        padding = torch.zeros(target_state_size - current_dim)
        state_vector_flat = torch.cat([state_vector_flat, padding])
    elif current_dim > target_state_size:
        state_vector_flat = state_vector_flat[:target_state_size]

    return state_vector_flat.unsqueeze(0)


class QDecayingDQNAgent:
    def __init__(self, state_size, action_size_total_discrete):
        self.state_size = state_size
        self.action_size = action_size_total_discrete
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.lr = LR_RICS

        self.model = OriginalDQN(state_size, action_size_total_discrete)
        self.target_model = OriginalDQN(state_size, action_size_total_discrete)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state_list_complex, action_idx, reward, next_state_list_complex, done):
        """Store an experience in the replay memory."""
        self.memory.add(state_list_complex, action_idx, reward, next_state_list_complex, done)

    def _process_state(self, state_list_complex_single):
        return _process_state_helper(state_list_complex_single, self.state_size)

    def act(self, state_list_complex_rics):
        """
        Select an action based on epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            action_idx = np.random.randint(self.action_size)
        else:
            state_tensor = self._process_state(state_list_complex_rics)
            with torch.no_grad():
                act_values = self.model(state_tensor)
            action_idx = torch.argmax(act_values).item()

        actual_phases = []
        temp_idx = action_idx
        choices_per_group = (2 ** FENBIAN)
        for _ in range(LL):
            choice_in_group = temp_idx % choices_per_group
            actual_phases.append(int(2 * np.pi * choice_in_group / choices_per_group))
            temp_idx //= choices_per_group
        actual_phases = actual_phases[::-1]

        return actual_phases, action_idx

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        if minibatch is None:
            return 0.0

        states_complex_b_list = [item[0] for item in minibatch]
        action_idx_b = np.array([item[1] for item in minibatch])
        rewards_b = np.array([item[2] for item in minibatch])
        next_states_complex_b_list = [item[3] for item in minibatch]
        dones_b = np.array([item[4] for item in minibatch])

        processed_states_b = torch.cat([self._process_state(s_list) for s_list in states_complex_b_list])
        processed_next_states_b = torch.cat([self._process_state(ns_list) for ns_list in next_states_complex_b_list])

        rewards_t = torch.tensor(rewards_b, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones_b, dtype=torch.float32).unsqueeze(1)
        action_idx_t = torch.tensor(action_idx_b, dtype=torch.long).unsqueeze(1)

        with torch.no_grad():
            target_q_next = self.target_model(processed_next_states_b)
            max_target_q_next, _ = torch.max(target_q_next, dim=1, keepdim=True)
            y_i = rewards_t + self.gamma * max_target_q_next * (1 - dones_t)

        current_q = self.model(processed_states_b)
        q_taken_action = current_q.gather(1, action_idx_t)

        loss = self.criterion(q_taken_action, y_i)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, name):
        """Load model weights."""
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()
        print(f"RICS DQN model loaded from {name}")

    def save(self, name):
        """Save model weights."""
        torch.save(self.model.state_dict(), name)
        print(f"RICS DQN model saved to {name}")


class MPDQNAgent:
    def __init__(self, state_size, num_discrete_actions, continuous_action_dim, n_v2v_links_for_alpha):
        self.state_size = state_size
        self.num_discrete_actions = num_discrete_actions
        self.continuous_action_dim = continuous_action_dim
        self.n_v2v_links = n_v2v_links_for_alpha

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.actor = ActorCV(state_size, num_discrete_actions, continuous_action_dim)
        self.target_actor = ActorCV(state_size, num_discrete_actions, continuous_action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR_CV)

        self.critic_state_feature_extractor = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.target_critic_state_feature_extractor = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.critic = CriticCV(state_features_dim=64, num_discrete_actions=num_discrete_actions,
                               continuous_action_dim=continuous_action_dim)
        self.target_critic = CriticCV(state_features_dim=64, num_discrete_actions=num_discrete_actions,
                                      continuous_action_dim=continuous_action_dim)

        critic_params = list(self.critic.parameters()) + list(self.critic_state_feature_extractor.parameters())
        self.critic_optimizer = optim.Adam(critic_params, lr=LR_CRITIC_CV)

        self.update_target_model()
        self.tau = TAU_PDQN

    def update_target_model(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic_state_feature_extractor.load_state_dict(self.critic_state_feature_extractor.state_dict())

    def _update_target_models_soft(self):
        """Soft update target networks using TAU."""
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic_state_feature_extractor.parameters(),
                                             self.critic_state_feature_extractor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state_list_complex, action_tuple, reward, next_state_list_complex, done):
        """Store an experience in the replay memory."""
        self.memory.add(state_list_complex, action_tuple, reward, next_state_list_complex, done)

    def _process_state(self, state_list_complex_single):
        return _process_state_helper(state_list_complex_single, self.state_size)

    def act(self, state_list_complex_cv):
        """
        Select an action
        """
        if np.random.rand() <= self.epsilon:
            discrete_action_idx = np.random.randint(self.num_discrete_actions)
            continuous_action_param_taken = np.random.rand(self.continuous_action_dim)
        else:
            state_tensor = self._process_state(state_list_complex_cv)
            self.actor.eval(); self.critic.eval(); self.critic_state_feature_extractor.eval()
            with torch.no_grad():
                all_continuous_params_actor = self.actor(state_tensor)

                q_values_for_all_discrete = []
                state_features = self.critic_state_feature_extractor(state_tensor)
                for i in range(self.num_discrete_actions):
                    discrete_action_one_hot = F.one_hot(torch.tensor([i], dtype=torch.long),
                                                        num_classes=self.num_discrete_actions).float()
                    continuous_param_for_this_disc = all_continuous_params_actor[:, i, :]
                    q_value = self.critic(state_features, discrete_action_one_hot, continuous_param_for_this_disc)
                    q_values_for_all_discrete.append(q_value)

                q_values_for_all_discrete = torch.cat(q_values_for_all_discrete, dim=1)

                discrete_action_idx = torch.argmax(q_values_for_all_discrete, dim=1).item()
                continuous_action_param_taken = all_continuous_params_actor[0, discrete_action_idx, :].cpu().numpy()
            self.actor.train(); self.critic.train(); self.critic_state_feature_extractor.train()

        alpha_actual = np.zeros(self.n_v2v_links, dtype=int)
        if discrete_action_idx < self.n_v2v_links:
            alpha_actual[discrete_action_idx] = 1
        rho_actual = float(continuous_action_param_taken[0])

        action_for_env = [alpha_actual.tolist(), rho_actual]
        action_tuple_for_replay = (discrete_action_idx, continuous_action_param_taken)

        return action_for_env, action_tuple_for_replay

    def replay(self, batch_size):
        """
        Train the MP-DQN Actor and Critic models
        """
        minibatch = self.memory.sample(batch_size)
        if minibatch is None:
            return 0.0

        states_complex_b_list = [item[0] for item in minibatch]
        action_tuples_b = [item[1] for item in minibatch]
        rewards_b = np.array([item[2] for item in minibatch])
        next_states_complex_b_list = [item[3] for item in minibatch]
        dones_b = np.array([item[4] for item in minibatch])

        processed_states_b = torch.cat([self._process_state(s_list) for s_list in states_complex_b_list])
        processed_next_states_b = torch.cat([self._process_state(ns_list) for ns_list in next_states_complex_b_list])

        discrete_actions_idx_b_np = np.array([a_tuple[0] for a_tuple in action_tuples_b])
        continuous_params_taken_b_np = np.array([a_tuple[1] for a_tuple in action_tuples_b])

        discrete_actions_idx_b_tensor = torch.tensor(discrete_actions_idx_b_np, dtype=torch.long)
        continuous_params_taken_b_tensor = torch.tensor(continuous_params_taken_b_np, dtype=torch.float32)

        rewards_b_tensor = torch.tensor(rewards_b, dtype=torch.float32).unsqueeze(1)
        dones_b_tensor = torch.tensor(dones_b, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            next_state_features_target = self.target_critic_state_feature_extractor(processed_next_states_b)
            next_all_continuous_params_target = self.target_actor(processed_next_states_b)

            max_next_q_values = torch.full((batch_size, 1), float('-inf'))
            for i in range(self.num_discrete_actions):
                discrete_action_one_hot_b = F.one_hot(torch.full((batch_size,), i, dtype=torch.long),
                                                      num_classes=self.num_discrete_actions).float()
                continuous_param_for_this_disc_b = next_all_continuous_params_target[:, i, :]

                q_values_current_disc = self.target_critic(next_state_features_target, discrete_action_one_hot_b,
                                                           continuous_param_for_this_disc_b)
                max_next_q_values = torch.max(max_next_q_values, q_values_current_disc)

            target_q_for_critic_update = rewards_b_tensor + (self.gamma * max_next_q_values * (1 - dones_b_tensor))

        current_state_features = self.critic_state_feature_extractor(processed_states_b)
        discrete_actions_one_hot_b = F.one_hot(discrete_actions_idx_b_tensor,
                                               num_classes=self.num_discrete_actions).float()

        current_q_for_taken_action = self.critic(current_state_features, discrete_actions_one_hot_b,
                                                 continuous_params_taken_b_tensor)

        critic_loss = F.mse_loss(current_q_for_taken_action, target_q_for_critic_update)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        actor_generated_all_params = self.actor(processed_states_b)

        actor_loss_components = []
        for i in range(self.num_discrete_actions):
            discrete_action_one_hot_b = F.one_hot(torch.full((batch_size,), i, dtype=torch.long),
                                                  num_classes=self.num_discrete_actions).float()
            continuous_param_for_this_disc_b = actor_generated_all_params[:, i, :]

            q_value_from_actor_params = self.critic(current_state_features.detach(),
                                                    discrete_action_one_hot_b,
                                                    continuous_param_for_this_disc_b)
            actor_loss_components.append(-q_value_from_actor_params.mean())

        actor_loss = torch.stack(actor_loss_components).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._update_target_models_soft()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return critic_loss.item()

    def load(self, name_prefix):
        """Load MP-DQN model weights from files."""
        self.actor.load_state_dict(torch.load(f"{name_prefix}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{name_prefix}_critic.pth"))
        self.critic_state_feature_extractor.load_state_dict(torch.load(f"{name_prefix}_critic_features.pth"))
        self.update_target_model()
        print(f"MP-DQN models loaded from prefix {name_prefix}")

    def save(self, name_prefix):
        """Save MP-DQN model weights to files."""
        torch.save(self.actor.state_dict(), f"{name_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{name_prefix}_critic.pth")
        torch.save(self.critic_state_feature_extractor.state_dict(), f"{name_prefix}_critic_features.pth")
        print(f"MP-DQN models saved with prefix {name_prefix}")