import numpy as np

EPISODES = 600
MAX_STEPS = 200
BATCH_SIZE = 32
GAMMA = 0.95  # Used by agents
LR_RICS = 0.001  # LR for RICS DQN Agent
EPSILON = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01
MEMORY_SIZE = 5000
TARGET_UPDATE = 20

# Environment/RICS Parameters
# L is the number of RICS elements
L = 30
# LL is the number of RICS sub-surfaces for phase control
LL = 2
# fenbian is the resolution bits for phase shifts
FENBIAN = 2

# K_NUM_RICS: Number of RICS units
K_NUM_RICS = 1

# M_NUM_CV: Number of AVs
M_NUM_CV = 10
# N_NUM_V2V: Number of V2V pairs
N_NUM_V2V = 2

# --- MP-DQN Hyperparameters ---
LR_ACTOR_CV = 0.0001
LR_CRITIC_CV = 0.001
TAU_PDQN = 0.005

# CV action space details
CV_NUM_DISCRETE_ACTIONS = N_NUM_V2V + 1
CV_CONTINUOUS_ACTION_DIM = 1  # Offloading ratio rho

# For RICS Agent (QDecayingDQNAgent)
STATE_SIZE_RIS = 2 * (L * M_NUM_CV + L * N_NUM_V2V + K_NUM_RICS * L)

# For CV Agent (MPDQNAgent)
STATE_SIZE_CV = 2 * (M_NUM_CV * N_NUM_V2V + M_NUM_CV * K_NUM_RICS)

# --- Channel Model Parameters ---
a_bm = 3
a_rm = 2.2
a_br = 2.5
a_Rn = 2.3
k_const1 = 3 / 4
k_const2 = 1 / 3

# --- System Parameters ---
PM_TX_POWER_DBM = 29
PT_TX_POWER_DBM = 22
NOISE_POWER_DBM = -110

PM_TX_POWER_LINEAR = 10 ** ((PM_TX_POWER_DBM - 30) / 10)
PT_TX_POWER_LINEAR = 10 ** ((PT_TX_POWER_DBM - 30) / 10)
NOISE_POWER_LINEAR = 10 ** ((NOISE_POWER_DBM - 30) / 10)

AM_Q = 0.7
AB_Q = 0.8
F_EDGE_CPU_HZ = 50 * 1e9
BW_V2I_TOTAL = 10e6
BW_V2V_TOTAL = 10e6
GAMMA_TH_V2V = 2
C_PENALTY = -10
NORMALIZATION_FACTOR_RN = 1e-10