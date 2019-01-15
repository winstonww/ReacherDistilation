from pathlib import Path
from baselines.common.distributions import make_pdtype
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
import datetime
import os 
####################### GLOBAL DEFINES ########################
#                                                             #
###############################################################

# # number of steps to unroll 
# STEPS_UNROLLED = 20
# LSTM_BATCH_SIZE=100
# NUM_UNITS=100
 
DATE = datetime.datetime.now().strftime( "%Y%m%d" )
TIME = datetime.datetime.now().strftime( "%H%M%S" )
EPISODE_STEPS = 50
OBSPACE_SHAPE=11
ACSPACE_SHAPE=2
PDFLAT_SHAPE = 4
GAMMA = 0.99

# number of steps to unroll 
TOTAL_EPISODES=8000
STEPS_UNROLLED = 10
LSTM_BATCH_SIZE=20
#LSTM_BATCH_SIZE=20
MLP_BATCH_SIZE=20
NUM_UNITS=200
KEEP_PROB = 0.5
MAX_CAPACITY=10
TRAINING_EPOCHS=1

base_path =  "{0}/reacher/data".format( str(Path.home()) )
date_path = "{0}/{1}".format(base_path, DATE )
time_path = "{0}/{1}".format(date_path, TIME )
lstm_path = time_path + "/lstm"
if not (os.path.isdir(date_path)):
    os.mkdir(date_path)
if not (os.path.isdir(time_path)):
    os.mkdir(time_path)
if not (os.path.isdir(lstm_path)):
    os.mkdir(lstm_path)

lstm_trained_data_path = "{0}/lstm_with_keep_probability_{1}.ckpt".format( base_path, KEEP_PROB )
dataset_path = "{0}/dataset_kp_{1}".format( lstm_path, KEEP_PROB )
if not (os.path.isdir(dataset_path)):
    os.mkdir(dataset_path)
