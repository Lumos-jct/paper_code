import random
import numpy as np
from config import *

class RISD2DEnv:
    def __init__(self):
        self.RICS_position = (0, 0)
        self.BS_position = (20, 0)

        self.last_action_rics = np.zeros(L, dtype=np.float32)
        
        self.last_action_cvs = []
        for _ in range(M_NUM_CV):

            omega_init = np.zeros(N_NUM_V2V, dtype=np.float32) 
            rho_init = np.array([0.0], dtype=np.float32)
            self.last_action_cvs.append((omega_init, rho_init))

        self.current_position_cvs, self.current_position_v2vs = self.initialize_positions(M_NUM_CV, N_NUM_V2V)
        
        self.state_ris, self.state_cv = self.calculate_state(self.current_position_cvs, self.current_position_v2vs)

    def initialize_positions(self, num_cvs, num_v2vs):
        x_range_cv = range(250, 351)
        y_range_cv = range(-20, 21)
        position_cvs = [(random.choice(x_range_cv), random.choice(y_range_cv)) for _ in range(num_cvs)]
        x_range_v2v_tx = range(-250, 1)
        y_range_v2v_tx = range(-250, 1)
        position_v2v_tx = [(random.choice(x_range_v2v_tx), random.choice(y_range_v2v_tx)) for _ in range(num_v2vs)]
        position_v2vs_pairs = []
        for v2v_tx_pos in position_v2v_tx:
            x_new = v2v_tx_pos[0] + random.randint(-10, 10)
            y_new = v2v_tx_pos[1] + random.randint(-10, 10)
            position_v2vs_pairs.append((v2v_tx_pos, (x_new, y_new)))
        return position_cvs, position_v2vs_pairs

    def _caculate_new_positions_for_step(self):
        self.current_position_cvs = [(x + random.uniform(-1, 1) * 2, y + random.uniform(-1, 1) * 2) for x, y in
                                     self.current_position_cvs]
        new_v2vs = []
        for tx, rx in self.current_position_v2vs:
            new_tx = (tx[0] + random.uniform(-1, 1) * 2, tx[1] + random.uniform(-1, 1) * 2)
            new_rx = (rx[0] + random.uniform(-1, 1) * 2, rx[1] + random.uniform(-1, 1) * 2)
            new_v2vs.append((new_tx, new_rx))
        self.current_position_v2vs = new_v2vs

    def nlos(self, dist, a_exp):
        rho_pl_at_1m = 10 ** (-20 / 10)
        if dist <= 1e-6: dist = 1e-6
        gain = np.sqrt(rho_pl_at_1m) * (dist ** (-a_exp / 2.0))
        return gain

    def Rayle_model(self):
        return np.sqrt(1 / 2) * (np.random.randn() + 1j * np.random.randn())

    def hmr_los(self, dist):
        wavelen = 0.01
        phase = -2 * np.pi * dist / wavelen
        return np.exp(1j * phase)

    def initialize_channel_gains(self, pos_cvs_current, pos_v2vs_current):
        self.h_mR = np.zeros((L, M_NUM_CV), dtype=complex)
        dist_CV_RICS = np.linalg.norm(np.array(pos_cvs_current) - np.array(self.RICS_position), axis=1)
        for i in range(M_NUM_CV):
            pl_gain = self.nlos(dist_CV_RICS[i], a_rm)
            los_c = self.hmr_los(dist_CV_RICS[i]) * np.sqrt(k_const1 / (k_const1 + k_const2))
            nlos_c = self.Rayle_model() * np.sqrt(k_const2 / (k_const1 + k_const2))
            self.h_mR[:, i] = pl_gain * (los_c + nlos_c)

        self.h_Rn = np.zeros((L, N_NUM_V2V), dtype=complex)
        dist_RICS_V2V_Rx = np.array(
            [np.linalg.norm(np.array(p_v2v[1]) - np.array(self.RICS_position)) for p_v2v in pos_v2vs_current])
        for i in range(N_NUM_V2V):
            pl_gain = self.nlos(dist_RICS_V2V_Rx[i], a_Rn)
            los_c = self.hmr_los(dist_RICS_V2V_Rx[i]) * np.sqrt(k_const1 / (k_const1 + k_const2))
            nlos_c = self.Rayle_model() * np.sqrt(k_const2 / (k_const1 + k_const2))
            self.h_Rn[:, i] = pl_gain * (los_c + nlos_c)

        self.h_RB = np.zeros((1, L), dtype=complex)
        dist_RB = np.linalg.norm(np.array(self.RICS_position) - np.array(self.BS_position))
        pl_gain_rb = self.nlos(dist_RB, a_br)
        los_c_rb = self.hmr_los(dist_RB) * np.sqrt(k_const1 / (k_const1 + k_const2))
        nlos_c_rb = self.Rayle_model() * np.sqrt(k_const2 / (k_const1 + k_const2))
        self.h_RB[0, :] = pl_gain_rb * (los_c_rb + nlos_c_rb)

        self.h_nn = np.zeros((N_NUM_V2V, 1), dtype=complex)
        dist_V2V_direct = np.array(
            [np.linalg.norm(np.array(p_v2v[1]) - np.array(p_v2v[0])) for p_v2v in pos_v2vs_current])
        for i in range(N_NUM_V2V):
            self.h_nn[i, 0] = (self.nlos(dist_V2V_direct[i], 2.0) * self.Rayle_model())

        self.h_mB = np.zeros((M_NUM_CV, 1), dtype=complex)
        dist_BS_CV = np.linalg.norm(np.array(pos_cvs_current) - np.array(self.BS_position), axis=1)
        for i in range(M_NUM_CV):
            pl_gain = self.nlos(dist_BS_CV[i], a_bm)
            los_c = self.hmr_los(dist_BS_CV[i]) * np.sqrt(k_const1 / (k_const1 + k_const2))
            nlos_c = self.Rayle_model() * np.sqrt(k_const2 / (k_const1 + k_const2))
            self.h_mB[i, 0] = pl_gain * (los_c + nlos_c)
        
        self.h_nB = np.zeros((N_NUM_V2V, 1), dtype=complex)
        dist_BS_V2V_Tx = np.array(
            [np.linalg.norm(np.array(p_v2v[0]) - np.array(self.BS_position)) for p_v2v in pos_v2vs_current])
        for i in range(N_NUM_V2V):
            pl_gain = self.nlos(dist_BS_V2V_Tx[i], 2.0)
            los_c = self.hmr_los(dist_BS_V2V_Tx[i]) * np.sqrt(k_const1 / (k_const1 + k_const2))
            nlos_c = self.Rayle_model() * np.sqrt(k_const2 / (k_const1 + k_const2))
            self.h_nB[i, 0] = pl_gain * (los_c + nlos_c)

        self.h_mn = np.zeros((M_NUM_CV, N_NUM_V2V), dtype=complex)
        for i in range(M_NUM_CV):
            for j in range(N_NUM_V2V):
                dist_cv_v2v_rx = np.linalg.norm(np.array(pos_cvs_current[i]) - np.array(pos_v2vs_current[j][1]))
                self.h_mn[i, j] = (self.nlos(dist_cv_v2v_rx, 2.0) * self.Rayle_model())

    def calculate_state(self, pos_cvs, pos_v2vs):

        self.initialize_channel_gains(pos_cvs, pos_v2vs)
        
        state_ris = [
            self.last_action_rics.copy(), 
            self.h_mR.copy(),             
            self.h_Rn.copy(),            
            self.h_RB.copy()              
        ]
        
        
        state_cv_list = []
        for u in range(M_NUM_CV):
            powers = np.array([PM_TX_POWER_LINEAR, PT_TX_POWER_LINEAR], dtype=np.float32)
            
            omega_prev = self.last_action_cvs[u][0]
            rho_prev = self.last_action_cvs[u][1]   
            prev_action_vec = np.concatenate([omega_prev, rho_prev])
            
            h_uv_local = self.h_mn[u, :] 
            
            h_ub_local = self.h_mB[u, :]
            
            s_u = [
                powers,             
                prev_action_vec,     
                h_uv_local,          
                h_ub_local          
            ]
            state_cv_list.append(s_u)
            
        return state_ris, state_cv_list

    def reset(self):
        # Reset Actions
        self.last_action_rics = np.zeros(L, dtype=np.float32)
        self.last_action_cvs = []
        for _ in range(M_NUM_CV):
            omega_init = np.zeros(N_NUM_V2V, dtype=np.float32)
            rho_init = np.array([0.0], dtype=np.float32)
            self.last_action_cvs.append((omega_init, rho_init))
            
        self.current_position_cvs, self.current_position_v2vs = self.initialize_positions(M_NUM_CV, N_NUM_V2V)
        return self.calculate_state(self.current_position_cvs, self.current_position_v2vs)

    def step(self, action_ris_actual_phases, action_cv_list_for_env):
        reward, Sm_sum_val, Rn_sum_val = self.calculate_reward(action_ris_actual_phases, action_cv_list_for_env)

        phases_full = np.zeros(L, dtype=np.float32)
        elements_per_group = L // LL
        for i_group in range(LL):
            val = action_ris_actual_phases[i_group]
            start_idx = i_group * elements_per_group
            end_idx = start_idx + elements_per_group
            phases_full[start_idx:end_idx] = val
        self.last_action_rics = phases_full

        for u in range(M_NUM_CV):
            alpha_list = action_cv_list_for_env[u][0]
            rho_val = action_cv_list_for_env[u][1]    
            
            omega_vec = np.array(alpha_list, dtype=np.float32)
            rho_vec = np.array([rho_val], dtype=np.float32)
            
            self.last_action_cvs[u] = (omega_vec, rho_vec)

        self._caculate_new_positions_for_step()
        
        next_state_ris_list, next_state_cv_list = self.calculate_state(self.current_position_cvs,
                                                                      self.current_position_v2vs)

        done = np.random.rand() > 0.95
        return next_state_ris_list, next_state_cv_list, reward, Sm_sum_val, Rn_sum_val, done

    def check_rule(self, alpha_all_cvs_list):
        if not alpha_all_cvs_list: return True
        num_v2v_links = len(alpha_all_cvs_list[0])
        for j_v2v_idx in range(num_v2v_links):
            sum_sharing_v2v_link_j = sum(alpha_m[j_v2v_idx] for alpha_m in alpha_all_cvs_list)
            if sum_sharing_v2v_link_j > 1:
                return False
        return True

    def calculate_reward(self, action_ris_phases, action_cv_list_from_agents):
        Pm = PM_TX_POWER_LINEAR
        Pt = PT_TX_POWER_LINEAR
        W_sigma_sq = NOISE_POWER_LINEAR

        Theta_diag_V2I = np.zeros(L, dtype=complex)
        elements_per_group = L // LL
        for i_group in range(LL):
            phase_val = action_ris_phases[i_group]
            for j_elem_in_group in range(elements_per_group):
                element_idx = i_group * elements_per_group + j_elem_in_group
                if element_idx < L:
                    Theta_diag_V2I[element_idx] = np.exp(1j * phase_val) * BETA_r
        Theta_V2I_matrix = np.diag(Theta_diag_V2I)

        Theta_diag_V2V_interf = np.zeros(L, dtype=complex)
        for i_group in range(LL):
            phase_val_v2v = action_ris_phases[i_group] * PSI
            for j_elem_in_group in range(elements_per_group):
                element_idx = i_group * elements_per_group + j_elem_in_group
                if element_idx < L:
                    Theta_diag_V2V_interf[element_idx] = np.exp(1j * phase_val_v2v) * BETA_t
        Theta_V2V_interf_matrix = np.diag(Theta_diag_V2V_interf)

        alphas_list_of_lists = [item[0] for item in action_cv_list_from_agents]
        rhos_list_of_scalars = [item[1] for item in action_cv_list_from_agents]

        rule_ok = self.check_rule(alphas_list_of_lists)

        SINR_m_vals = np.zeros(M_NUM_CV)
        for i_cv in range(M_NUM_CV):
            h_d_v2i = self.h_mB[i_cv, 0]
            h_r_v2i = self.h_RB[0, :] @ Theta_V2I_matrix @ self.h_mR[:, i_cv]

            sig_v2i = Pm * np.abs(h_d_v2i + h_r_v2i) ** 2

            interf_v2i = 0
            alpha_this_cv = alphas_list_of_lists[i_cv]
            for j_v2v_link in range(N_NUM_V2V):
                if alpha_this_cv[j_v2v_link] == 1:
                    interf_v2i += Pt * np.abs(self.h_nB[j_v2v_link, 0]) ** 2

            denominator = interf_v2i + W_sigma_sq
            SINR_m_vals[i_cv] = sig_v2i / (denominator if denominator > 1e-12 else 1e-12)

        SINR_n_vals = np.zeros(N_NUM_V2V)
        for j_v2v_link in range(N_NUM_V2V):
            sig_v2v = Pt * np.abs(self.h_nn[j_v2v_link, 0]) ** 2

            interf_v2v = 0
            for i_cv_interferer in range(M_NUM_CV):
                alpha_interfering_cv = alphas_list_of_lists[i_cv_interferer]
                if alpha_interfering_cv[j_v2v_link] == 1:
                    direct_interf_path = self.h_mn[i_cv_interferer, j_v2v_link]
                    rics_interf_path = np.conjugate(self.h_Rn[:, j_v2v_link].T) @ Theta_V2V_interf_matrix @ self.h_mR[:,
                                                                                                            i_cv_interferer]
                    interf_v2v += Pm * np.abs(direct_interf_path + rics_interf_path) ** 2

            denominator = interf_v2v + W_sigma_sq
            SINR_n_vals[j_v2v_link] = sig_v2v / (denominator if denominator > 1e-12 else 1e-12)

        Sm_sum = 0
        
        sm_data_size_bits_per_task = np.random.uniform(3e6, 6e6, size=M_NUM_CV)
        cm_cycles_per_task = np.random.uniform(1e9, 5e9, size=M_NUM_CV)
        fm_local_cpu_hz = np.random.uniform(1e9, 5e9, size=M_NUM_CV)
        
        BW_V2I_per_CV = BW_V2I_TOTAL / M_NUM_CV if M_NUM_CV > 0 else BW_V2I_TOTAL

        for i_cv in range(M_NUM_CV):
            rho_this_cv = rhos_list_of_scalars[i_cv]

            T_local_i = (1 - rho_this_cv) * (cm_cycles_per_task[i_cv] / fm_local_cpu_hz[i_cv])
            
            rate_v2i_bps = BW_V2I_per_CV * np.log2(1 + SINR_m_vals[i_cv])
            if rate_v2i_bps <= 1e-9: rate_v2i_bps = 1e-9
            
            T_offload_tx_i = rho_this_cv * (sm_data_size_bits_per_task[i_cv] / rate_v2i_bps)
            T_offload_comp_i = rho_this_cv * (cm_cycles_per_task[i_cv] / F_EDGE_CPU_HZ)

            T_offload_total_i = T_offload_tx_i + T_offload_comp_i
            tm_i = max(T_local_i, T_offload_total_i)
            if tm_i <= 1e-9: tm_i = 1e-9

            current_acc_i = (1 - rho_this_cv) * AM_Q + rho_this_cv * AB_Q
            Sm_sum += current_acc_i / tm_i

        Rn_sum = 0
        BW_V2V_per_link = BW_V2V_TOTAL / N_NUM_V2V if N_NUM_V2V > 0 else BW_V2V_TOTAL
        for j_v2v_link in range(N_NUM_V2V):
            Rn_sum += BW_V2V_per_link * np.log2(1 + SINR_n_vals[j_v2v_link])

        constraint_penalty_sum = 0
        for sinr_n_val in SINR_n_vals:
            constraint_penalty_sum += min(sinr_n_val - GAMMA_TH_V2V, 0)

        reward = Sm_sum + constraint_penalty_sum + NORMALIZATION_FACTOR_RN * Rn_sum

        if not rule_ok:
            reward += C_PENALTY

        return reward, Sm_sum, Rn_sum
