import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from config import *
from zxy_code.agents import QDecayingDQNAgent, MPDQNAgent
from zxy_code.environment import RISD2DEnv

if __name__ == "__main__":
    env = RISD2DEnv()

    total_rics_discrete_actions = (2 ** FENBIAN) ** LL
    q_decaying_agent = QDecayingDQNAgent(STATE_SIZE_RIS, total_rics_discrete_actions)

    cv_agents_list = []
    for i in range(M_NUM_CV):
        agent = MPDQNAgent(STATE_SIZE_CV, CV_NUM_DISCRETE_ACTIONS, CV_CONTINUOUS_ACTION_DIM, N_NUM_V2V)
        cv_agents_list.append(agent)

    sm_history = []
    rn_history = []
    avg_sm_history = []
    episode_rewards_history = []

    start_time = time.time()
    for e in range(EPISODES):
        current_s_ris_list, current_s_cv_list = env.reset()

        episode_sm_sum = 0
        episode_rn_sum = 0
        episode_reward_sum = 0
        steps_this_episode = 0

        for t in range(MAX_STEPS):
            steps_this_episode += 1
            actual_rics_phases, rics_action_idx_replay = q_decaying_agent.act(current_s_ris_list)

            action_cv_for_env_list = []
            action_cv_for_replay_list = []

            for m_idx in range(M_NUM_CV):
                action_for_env_m, action_tuple_for_replay_m = cv_agents_list[m_idx].act(current_s_cv_list)
                action_cv_for_env_list.append(action_for_env_m)
                action_cv_for_replay_list.append(action_tuple_for_replay_m)

            next_s_ris_list, next_s_cv_list, reward_step, Sm_step, Rn_step, done_step = \
                env.step(actual_rics_phases, action_cv_for_env_list)

            q_decaying_agent.remember(current_s_ris_list, rics_action_idx_replay, reward_step, next_s_ris_list,
                                      done_step)
            for m_idx in range(M_NUM_CV):
                action_tuple_m_replay = action_cv_for_replay_list[m_idx]
                cv_agents_list[m_idx].remember(current_s_cv_list, action_tuple_m_replay, reward_step,
                                               next_s_cv_list, done_step)

            current_s_ris_list, current_s_cv_list = next_s_ris_list, next_s_cv_list
            episode_sm_sum += Sm_step
            episode_rn_sum += Rn_step
            episode_reward_sum += reward_step

            if done_step:
                break

            loss_q_val = q_decaying_agent.replay(BATCH_SIZE)

            for m_idx in range(M_NUM_CV):
                loss_cv_agent_m = cv_agents_list[m_idx].replay(BATCH_SIZE)

        avg_sm_episode = episode_sm_sum / steps_this_episode if steps_this_episode > 0 else 0
        avg_rn_episode = episode_rn_sum / steps_this_episode if steps_this_episode > 0 else 0

        avg_sm_episode = episode_sm_sum / steps_this_episode if steps_this_episode > 0 else 0
        avg_sm_history.append(avg_sm_episode)

        sm_history.append(avg_sm_episode)
        rn_history.append(avg_rn_episode)
        episode_rewards_history.append(episode_reward_sum)

        if (e + 1) % TARGET_UPDATE == 0:
            q_decaying_agent.update_target_model()
            for cv_agent_instance in cv_agents_list:
                cv_agent_instance.update_target_model()

        print(
            f"Ep: {e + 1}/{EPISODES}, Steps: {steps_this_episode}, AvgSm: {avg_sm_episode:.4f}, AvgRn: {avg_rn_episode:.2e}, EpRwd: {episode_reward_sum:.2f}, RICS_Eps: {q_decaying_agent.epsilon:.3f}, CV_Eps: {cv_agents_list[0].epsilon if M_NUM_CV > 0 else 0:.3f}")

    end_time = time.time()
    if len(avg_sm_history) >= 200:
        average_sm_last_200_episodes = np.mean(avg_sm_history[-200:])
    else:
        average_sm_last_200_episodes = np.mean(avg_sm_history)
    print(average_sm_last_200_episodes)
    print(f"Training completed in {end_time - start_time:.2f} seconds.")