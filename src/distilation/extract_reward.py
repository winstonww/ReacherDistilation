#!/usr/bin/env python
import sys
sys.path.append("/home/winstonww/reacher/src")
from distilation.config import *
from distilation.dataset import Dataset
import numpy as np
from json_tricks import loads
import json
import types


class ExtractReward(object):

    @classmethod
    def get_episode_reward(cls, episode):
        re = []
        for i in range(len(episode)):
            if isinstance(episode[i]["rew"], list): re.append( episode[i]["rew"][0] )
            else: re.append( episode[i]["rew"] )
        return re

    @classmethod
    def get_return(cls, dataset):
        ret = []
        pages = dataset.pages()
        for i in range(len(pages)):
            dataset.switch(pages[i])
            data = dataset.data_in_memory
            for i in range(len(data)):
                ret.append( sum( cls.get_episode_reward(data[i]) ) )
        return ret

    @classmethod
    def get_avg_return(cls, dataset, per_episodes):
        avg_ret = []
        ret = cls.get_return(dataset)
        for i in range(0, len(ret), per_episodes):
            end = min( i+per_episodes, len(ret) )
            avg_ret.append( sum( ret[i:end] ) / len( ret[i:end] ) )
        return avg_ret
    
    @classmethod
    def get_avg_reward(cls, dataset, per_episodes):
        avg_rew = []
        avg_ret = cls.get_avg_return( dataset, per_episodes)
        for i in range(len(avg_ret)):
            avg_rew.append( avg_ret[i] / EPISODE_STEPS )
        return avg_rew
    

# kp=1
path = "/home/winstonww/reacher/data/20181223/065315/lstm/dataset_kp_1"

## 0.9, normal num units = 50. training epoch = 15
#path = "/home/winstonww/reacher/data/20181223/174324/lstm/dataset_kp_0.9"
#
##0.9 prev_array = 0. lol works better than ^ wtf
#path = "/home/winstonww/reacher/data/20181224/071937/lstm/dataset_kp_0.9"
#
##0.9 feed rand to prev_array
#path = "/home/winstonww/reacher/data/20181224/203029/lstm/dataset_kp_0.9"
#
##0.9 normal, num_units = 500, training epoch 1
#path = "/home/winstonww/reacher/data/20181225/002556/lstm/dataset_kp_0.9"
#
## 0.5kp, prev_pdflat_batch=np.zero
#path = "/home/winstonww/reacher/data/20181225/223430/lstm/dataset_kp_0.5"
#
##0.5kp; correct prev_pdflat
#path = "/home/winstonww/reacher/data/20181226/035925/lstm/dataset_kp_0.5"
#
##0.3kp; prev_pdflat=ob
path = "/home/winstonww/reacher/data/20181227/031415/lstm/dataset_kp_0.3"

## 0.0kp; prev_pdflat=prev_pdflat
path = "/home/winstonww/reacher/data/20181227/230237/lstm/dataset_kp_0.0"

## 0.0kp; prev_pdflat=rand
path = "/home/winstonww/reacher/data/20181230/004313/lstm/dataset_kp_0.0"

# 0.0kp mlp, remove activation in last layer in mlp; prev_pdflat=prev_pdflat
path = "/home/winstonww/reacher/data/20190101/054413/mlp/dataset_kp_0.0"

#0.0kp mlp; add reward to input; prev_pdflat=prev_pdflat
#path = "/home/winstonww/reacher/data/20190101/170327/mlp/dataset_kp_0.0"

#0.7kp mlp: 1e-4 adam rate
path = "/home/winstonww/reacher/data/20190101/214853/mlp/dataset_kp_0.7"

#1.0kp mlp
path = "/home/winstonww/reacher/data/20190102/002714/mlp/dataset_kp_1"

#0.9 mlp zero rew zero prev_pdflat
path = "/home/winstonww/reacher/data/20190102/011959/mlp/dataset_kp_0.9"

#0.9 mlp normal
#path = "/home/winstonww/reacher/data/20190102/012252/mlp/dataset_kp_0.9"

#0.9 rand 
#path = "/home/winstonww/reacher/data/20190102/043730/mlp/dataset_kp_0.9"

#0.9 lstm normal 
path = "/home/winstonww/reacher/data/20190105/204301/lstm/dataset_kp_0.9"

#0.9
#path = "/home/winstonww/reacher/data/20190106/044645/lstm/dataset_kp_0.9"
path = "/home/winstonww/reacher/data/20190106/044645/lstm/dataset_kp_0.9"

#1.0
path = "/home/winstonww/reacher/data/20190106/052902/lstm/dataset_kp_1"

#0.95 without action
path = "/home/winstonww/reacher/data/20190106/061304/lstm/dataset_kp_0.95"

#path with action
path = "/home/winstonww/reacher/data/20190106/061154/lstm/dataset_kp_0.95"


### adjusted training epoch #####
path = "/home/winstonww/reacher/data/20190106/160559/lstm/dataset_kp_0.95"

# SUCCESS. 
# NUM_UNITS=100 
# NN:
#    for i in range(STEPS_UNROLLED):
#
#        dense, state_batch = cell( training_input_batch[ i, :, : ], state_batch )
#        dense = tf.nn.tanh( tf.layers.dense(dense, 64 ) )
#        dense = tf.nn.tanh( tf.layers.dense(dense, 128 ) )
#        dense = tf.nn.tanh( tf.layers.dense(dense, 64 ) )
#        dense = tf.nn.tanh( tf.layers.dense(dense, 32 ) )
#        dense = tf.layers.dense(dense, PDFLAT_SHAPE )
#        output_batch.append(dense)
#TRAINING EPOCH = 1
#LSTM_BATCH_SIZE=20
#SWITCHING PAGE EVERY 20 
path = "/home/winstonww/reacher/data/20190106/162053/lstm/dataset_kp_0.85"


# no action fed 0.85 kp
path = "/home/winstonww/reacher/data/20190106/165610/lstm/dataset_kp_0.85"


# no action
path = "/home/winstonww/reacher/data/20190106/223013/lstm/dataset_kp_0.9"

#0.9 with prev action and random
path= "/home/winstonww/reacher/data/20190106/223013/lstm/dataset_kp_0.9"
path = "/home/winstonww/reacher/data/20190107/004224/lstm/dataset_kp_0.9"

#1.0 zero prev_pdflat
#path = "/home/winstonww/reacher/data/20190107/045651/lstm/dataset_kp_1"

#1.0 no prev_pdflat
path = "/home/winstonww/reacher/data/20190108/064703/lstm/dataset_kp_1"


#1.0 modify back prop
path = "/home/winstonww/reacher/data/20190109/224132/lstm/dataset_kp_1"

#1.0 remove prev
path = "/home/winstonww/reacher/data/20190110/042131/lstm/dataset_kp_1"

#1.0 set prev=0
path = "/home/winstonww/reacher/data/20190110/050910/lstm/dataset_kp_1"

#1.0 prev=prev
path = "/home/winstonww/reacher/data/20190110/054500/lstm/dataset_kp_1"
rets_path = "/home/winstonww/reacher/data/kp1.0"

#YAYYYYYYYYYYY!!!!!YASSSSS!!!!!!
#0.85 prev=prev
path = "/home/winstonww/reacher/data/20190110/170351/lstm/dataset_kp_0.85"
rets_path = "/home/winstonww/reacher/data/kp0.85"

# try 0.75 prev-prev
#path = "/home/winstonww/reacher/data/20190110/215203/lstm/dataset_kp_0.75"
#rets_path = "/home/winstonww/reacher/data/kp0.75"

# 0.5
#path = "/home/winstonww/reacher/data/20190110/225036/lstm/dataset_kp_0.5"
#rets_path = "/home/winstonww/reacher/data/kp0.5"

#0.0
#path = "/home/winstonww/reacher/data/20190110/230709/lstm/dataset_kp_0.0"
#rets_path = "/home/winstonww/reacher/data/kp0.0"

#0.2
#path = "/home/winstonww/reacher/data/20190113/195908/lstm/dataset_kp_0.2"
#rets_path = "/home/winstonww/reacher/data/kp0.2"


##################################OOPS######################################

#0.85

files = [
    ("/home/winstonww/reacher/data/20190113/203457/lstm/dataset_kp_0.85",
     "/home/winstonww/reacher/data/kp0.85" ),

#0.75
    ( "/home/winstonww/reacher/data/20190113/204535/lstm/dataset_kp_0.75",
    "/home/winstonww/reacher/data/kp0.75"),

#0.5
    ("/home/winstonww/reacher/data/20190113/204429/lstm/dataset_kp_0.5/",
     "/home/winstonww/reacher/data/kp0.5"),

#0.2
    ("/home/winstonww/reacher/data/20190114/051054/lstm/dataset_kp_0.2",
    "/home/winstonww/reacher/data/kp0.2"),

#0.1
    ("/home/winstonww/reacher/data/20190114/052659/lstm/dataset_kp_0.1",
    "/home/winstonww/reacher/data/kp0.1"),

#0.05 
    ("/home/winstonww/reacher/data/20190114/053052/lstm/dataset_kp_0.05",
    "/home/winstonww/reacher/data/kp0.05"),

#0.0
    ("/home/winstonww/reacher/data/20190114/052810/lstm/dataset_kp_0.0",
     "/home/winstonww/reacher/data/kp0.0"),

#1.0
    ("home/winstonww/reacher/data/20190114/054249/lstm/dataset_kp_1.0",
    "/home/winstonww/reacher/data/kp1.0"),
]

files = [
        #1.0
    ("/home/winstonww/reacher/data/20190114/054249/lstm/dataset_kp_1.0",
    "/home/winstonww/reacher/data/kp1.0"),
]

#step with student 0.75 kp
files = [
  ("/home/winstonww/reacher/data/20190114/055010/lstm/dataset_kp_0.75",
    "/home/winstonww/reacher/data/kp0.75s"),
  ]

files = [
        ("/home/winstonww/reacher/data/20190114/192136/lstm/dataset_kp_0.5",
        "/home/winstonww/reacher/data/kp0.5ss"),
    ]

for (path, rews_path) in files:
    dataset = Dataset(path)
    avg_rews = ExtractReward.get_avg_reward(dataset, 50)
    print(avg_rews)
    np.save(rews_path, avg_rews) 
    print("file written to {0}.npy, avg_rews array length is {1}".format(rews_path, len(avg_rews)))
    #print(json.dumps(dataset.data_in_memory[-1], indent=4, sort_keys=True))
