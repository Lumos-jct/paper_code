import torch
import torch.nn as nn
import torch.nn.functional as F

class OriginalDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(OriginalDQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.output_layer(x)

class ActorCV(nn.Module):
    def __init__(self, state_size, num_discrete_actions, continuous_action_dim):
        super(ActorCV, self).__init__()
        self.num_discrete_actions = num_discrete_actions
        self.continuous_action_dim = continuous_action_dim

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, num_discrete_actions * continuous_action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        continuous_params_flat = torch.sigmoid(self.output_layer(x))
        return continuous_params_flat.view(-1, self.num_discrete_actions, self.continuous_action_dim)

class CriticCV(nn.Module):
    def __init__(self, state_features_dim, num_discrete_actions, continuous_action_dim):
        super(CriticCV, self).__init__()

        self.fc_combined = nn.Linear(state_features_dim + num_discrete_actions + continuous_action_dim, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, state_features, discrete_action_one_hot, continuous_action_param):
        combined_input = torch.cat([state_features, discrete_action_one_hot, continuous_action_param], dim=1)
        x = F.relu(self.fc_combined(combined_input))
        return self.output_layer(x)