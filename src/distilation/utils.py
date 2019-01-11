import os
import numpy as np
#################$####### UTILS ###############################
#                                                             #
###############################################################
def load_file(path):
    arr = []
    if not os.path.exists(path):
        print("Source npy does not exists")
        # warnings.warn("Source npy does not exists")
    else:
        arr = np.load(path)
    return arr

