import numpy as np
from collections   import defaultdict
data = np.load('/home/sofa/SOFA_v23.06.00/bin/lapgym/sofa_env/sofa_env/scenes/soft_body_manipulation/trajectories/SoftBodyManipulationEnv_7/trajectory_dict.npy', allow_pickle=True)
print(data[[0]])
