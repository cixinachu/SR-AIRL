FRAME_STATE_DIM = 16
HISTORY_LEN = 11
DT = 0.1
GAMMA = 0.99

# NGSIMEnv.normalize_obs() normalization.
POS_LAT_SCALE = 25.0
POS_LONG_SCALE = 500.0
VEL_SCALE = 80.0
VEL_MIN = -40.0
VEL_OFFSET = VEL_MIN  # compatibility alias

# Shared dynamic anisotropic interaction kernel.
L_LONG = 15.0
L_LAT = 3.5
DIR_LAMBDA = 0.3
PRED_HORIZON = 1.0
PRED_STEPS = 11

ALPHA_INIT = 0.1

# Counterfactual preference supervision.
RANK_LAMBDA = 0.1
RANK_TEMPERATURE = 0.1
RANK_BATCH_SIZE = 16
RANK_WINDOW = 20

MAX_EPISODES = 15000
SEED = 42

