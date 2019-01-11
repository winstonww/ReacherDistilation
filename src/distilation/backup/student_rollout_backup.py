#!/usr/bin/env python
import os
import glob
import tensorflow as tf 
from baselines.common import tf_util as U
from baselines.common import Dataset
from tensorflow.python.tools import inspect_checkpoint as chkp
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.common.mpi_adam import MpiAdam
from baselines.common.distributions import make_pdtype
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from random import randint

import datetime
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

from tensorflow.python import pywrap_tensorflow



####################### GLOBAL DEFINES ########################
#                                                             #
###############################################################

TRAINING_BATCH_SIZE = 2000
TOTAL_EPISODES = 40000
TIMESTEPS_PER_EPISODE = 50
SLIDING_WINDOW = 200

# number of steps to unroll 
STEPS_UNROLLED = 20
LSTM_BATCH_SIZE=100
NUM_UNITS=100

# preward prediction training
TRAIN_REWARD_BATCH_SIZE = 50

######################## AGENTS ###############################
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

######################## AGENTS ###############################
#                                                             #
###############################################################

class TeacherAgent(object):
    def __init__(self,env,sess,restore, batch=TRAINING_BATCH_SIZE):
      self.pi = mlp_policy.MlpPolicy(name='pi',
              ob_space=env.observation_space, ac_space=env.action_space,
              hid_size=64, num_hid_layers=2, training_batch_size=batch)
      self.saver = tf.train.Saver(
          var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi'))
      if restore:
          self.saver.restore(sess, "{0}/teacher.ckpt".format(base_path))

class StudentAgent(object):
    def __init__(self,env,sess,restore,klts):
      self.pi = mlp_policy.MlpPolicy(name="s_pi_{0}".format("klts" if klts else "klst"), 
              ob_space=env.observation_space, ac_space=env.action_space,
              hid_size=64, num_hid_layers=2, training_batch_size=TRAINING_BATCH_SIZE, gaussian_fixed_var=False )
      self.saver = tf.train.Saver(
          var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="s_pi_{0}".format("klts" if klts else "klst")))
      if restore:
          self.saver.restore(sess, "{0}/student_{1}.ckpt".format(base_path, "klts" if klts else "klst"))

######################## REPLAY ###############################
#                                                             #
###############################################################

def teacher_replay():
    env = make_mujoco_env("Reacher-v2", 0)
    rets = []
    episode = 0
    ret = 0
    timestep = 1
    with tf.Session() as sess:
        teacher = TeacherAgent(env,sess,True)
        ob = env.reset()
        ob = np.expand_dims( ob, axis=0 )
        print( ob )
        while episode < TOTAL_EPISODES:
            timestep += 1
            ac, _ = teacher.pi.act(False, ob)
            ob, reward, new, _ = env.step(ac)
            ob = np.expand_dims( ob, axis=0 )
            ret += reward
            if new:
                print("********** Episode {0}, timestep {1} ***********".format(episode, timestep))
                print("return: {0}".format(reward))
                ob = env.reset()
                ob = np.expand_dims( ob, axis=0 )
                rets.append(ret)
                ret = 0
                timestep = 1
                episode += 1
            env.render()

    # save_results
    np.save(teacher_ret_path, rets) 

######################### LSTM ################################
#                                                             #
###############################################################

# This method constructs the lstm graph.
# The method takes in ob_combined, the stacked observations tensor,
# and returns stacked action mean, logstd and std of the student, and the final state.
def lstm_graph( ob_combined, input_state_combined, env ):

    # parse action distribution mean, logstd
    # get action space type
    pdtype = make_pdtype( env.action_space )

    # new cell
    cell = tf.contrib.rnn.LSTMCell( num_units=NUM_UNITS, name="lol" )

    # Initailize state with zero of batch size 1 and type float32
    #c_state, m_state = tf.split(input_state_combined, [1, 1], 0 ) 
    c_state, m_state = input_state_combined[0,:,:], input_state_combined[1,:,:]
    state = tf.tuple( [ c_state, m_state ] )

    s_mean_list, s_std_list, s_logstd_list = [], [], []

    for i in range(STEPS_UNROLLED):

        if i > 0: tf.get_variable_scope().reuse_variables()
        # normalize observation vector with rms
        rms = RunningMeanStd( shape=env.observation_space.shape )
        # only first step; the rest will need all (all batch, all observation space dim )
        obz = tf.clip_by_value( (ob_combined[i, :, :] - rms.mean) / rms.std, -5.0, 5.0 )
        output, state = cell( obz, state )

        output = tf.nn.tanh(
                tf.layers.dense(output, 64, name='last',
                kernel_initializer=U.normc_initializer(1.0) ) )

        # feed the output of lstm to a final FC layer
        # this 'flat' vector will be split into the mean and std of a pd
        pdparam = tf.layers.dense( output, pdtype.param_shape()[0], 
                name='final', kernel_initializer=U.normc_initializer( 0.01 ) )

        pd = pdtype.pdfromflat( pdparam )

        s_mean_list.append( pd.mean )
        s_std_list.append( pd.std )
        s_logstd_list.append( pd.logstd )

    # stack the outputs at each cell together so that we can conveniently compute loss and etc
    s_mean_combined = tf.stack( s_mean_list )
    s_std_combined = tf.stack( s_std_list )
    s_logstd_combined = tf.stack( s_logstd_list )
    final_state_combined = tf.stack( state )

    return s_mean_combined, s_std_combined, s_logstd_combined, final_state_combined


# This method compute this total loss 
def lstm_loss( s_mean_combined, s_std_combined, s_logstd_combined, 
        t_mean_combined, t_std_combined, t_logstd_combined, env ):
    
    with tf.variable_scope("LSTM"):
        return tf.reduce_sum(t_logstd_combined - s_logstd_combined + 
                (tf.square(s_std_combined) + tf.square(s_mean_combined - t_mean_combined)) / (2.0 * tf.square(t_std_combined)) - 0.5, axis=[2,1,0] )

def lstm_train(train, keep, lstm_trained_data_path, restore ):

    # Initialize the environment
    env = make_mujoco_env("Reacher-v2", 0)

    # new session
    sess = tf.Session()

    # initialize teacher agent
    teacher = TeacherAgent( env, sess, True, batch = 1 )


    # This observation placeholder is for querying teacher action
    ob_ph = U.get_placeholder( name="ob", dtype=tf.float32, 
          shape=[ 1, env.observation_space.shape[0] ] )

    ################################### For training LSTM student Agent ##########################################

    with tf.variable_scope("LSTM"):

        # this tf placeholder holds a batch of observations for lstm training
        ob_combined_ph = tf.placeholder(name="ob_combined_ph", dtype=tf.float32, 
                          shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.observation_space.shape[0] ] )

        # keep probability
        keep_prob = tf.placeholder(name="keep_prob",dtype=tf.float32, shape=[] ) 

        # apply dropout
        ob_combined = tf.nn.dropout( ob_combined_ph, keep_prob )


        # input state placeholder, outer dim is 2 because of c_state and m state
        input_state_combined = tf.placeholder( shape=[ 2, LSTM_BATCH_SIZE, NUM_UNITS ], dtype=tf.float32 )
        
        # state variables for lstm
        zero_state = np.zeros( [ 2, LSTM_BATCH_SIZE, NUM_UNITS ] )
        curr_state = zero_state

        # lstm graph; shape of s_mean_combined is still [STEPS_UNROLLED, LSTM_BATCH_SIZE, 2]
        s_mean_combined, s_std_combined, s_logstd_combined, final_state_combined  = lstm_graph( ob_combined, input_state_combined, env )

        # placeholder for teacher action paramters, shape is [5, 1, 2]
        t_mean_combined = tf.placeholder( name="t_mean_combined",     shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.action_space.shape[0] ], dtype=tf.float32 )
        t_std_combined = tf.placeholder( name="t_std_combined",       shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.action_space.shape[0] ], dtype=tf.float32 )
        t_logstd_combined = tf.placeholder( name="t_logstd_combined", shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.action_space.shape[0] ], dtype=tf.float32 )
        
        # get the action wrt last observation
        # beginning at last index (i.e. STEPS_UNROLLED-1 ), sample down 1 element in the first (outer)  dimension
        # , and all elements in the inner dimensions
        s_ac = tf.slice( s_mean_combined, [ ( STEPS_UNROLLED - 1 ), ( LSTM_BATCH_SIZE - 1 ),  0 ], [ 1, 1, -1 ] )

        # get kl loss 
        loss = lstm_loss( s_mean_combined, s_std_combined, s_logstd_combined, 
                t_mean_combined, t_std_combined, t_logstd_combined, env )

    # get a collection of students within the 'LSTM' scope for optimization
    student_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="LSTM" )

    ################################################################################################################

    ########################################### For reward prediction ##############################################

    with tf.variable_scope("reward_pred"):
        # the observation that agent see to produce next action
        prev_ob = tf.placeholder( name="prev_ob", dtype=tf.float32, shape=[ TRAIN_REWARD_BATCH_SIZE, env.observation_space.shape[0] ]  )
        
        # The action that agent see after seing prev_ob
        next_ac = tf.placeholder( name="next_ac", dtype=tf.float32, shape=[ TRAIN_REWARD_BATCH_SIZE, env.action_space.shape[0] ]  )

        # The reward after taking seeing prev_ob an taking action next_ac 
        reward_target = tf.placeholder( name="reward_target", dtype=tf.float32, shape=[ TRAIN_REWARD_BATCH_SIZE, ] )

        # model reward based on pervious observation and action corresponded to the observation
        flat = tf.concat( [ prev_ob, next_ac ], 1 )

        # first hidden layer of size ( batch_size, 64 ) 
        predicted_reward = tf.layers.dense( flat, 64, kernel_initializer=U.normc_initializer( 1.0 ), name="reward_hid_1" )

        # final output is of size ( batch, size, 1 )
        predicted_reward = tf.layers.dense( predicted_reward, 1, kernel_initializer=U.normc_initializer( 1.0 ), name="reward_out" )

        # use l2 loss the train reward 
        reward_l2_loss = tf.reduce_sum( tf.square( predicted_reward - reward_target ), axis=[ 1, 0 ] )
    
    # get a collection of students within the 'reward_pred' scope for optimization
    reward_pred_var = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="reward_pred" )

    ################################################################################################################

    # adam optimizer for minimize kl loss; learning rate is fixed here
    adam = tf.train.AdamOptimizer( learning_rate=1e-3,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8)
    
    minimize_adam = adam.minimize( loss, var_list=student_var )

    minimize_reward_l2 = adam.minimize( reward_l2_loss, var_list=reward_pred_var )


    # initializer; to be placed at the very end
    init = tf.variables_initializer( 
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LSTM") + 
            tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="reward_pred" ) )

    # keep track of env resets i.e. num of episodes
    num_resets = 0

    # saver for restoring/saving depending on whether or not to train
    saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM') + 
        tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="reward_pred" ) )

    # these lists are for 
    ob_list, t_mean_list, t_std_list, t_logstd_list, reward_list, stepped_action_list  = [], [], [], [], [], []

    # save loss kl and rets
    losses, rets, pred_rets = [], [], []
    ret, pred_ret = 0, 0

    with sess:

        # run initializer for lstm variables
        if not restore:
            sess.run(init)

        # run initializer
        sess.run( tf.variables_initializer( adam.variables() ) )

        # Initialize saver for training
        if restore: 
            if glob.glob( lstm_trained_data_path + "*" ):
                saver.restore( sess, lstm_trained_data_path )
            else:
                print( "attempt to restore trained data but {0} does not exist".format( lstm_trained_data_path ) )

            if os.path.exists( lstm_loss_path ):
                losses = load_file( lstm_loss_path ).tolist()
            else:
                print( "attempt to restore loss but {0} does not exist".format( lstm_trained_data_path ) )

            if os.path.exists( lstm_ret_path ):
                rets = load_file( lstm_ret_path ).tolist()
            else:
                print( "attempt to restore ret but {0} does not exist".format( lstm_ret_path ) )

            if os.path.exists( lstm_pred_ret_path ):
                pred_rets = load_file( lstm_pred_ret_path ).tolist()
            else:
                print( "attempt to restore pred ret but {0} does not exist".format( lstm_pred_ret_path ) )

            if os.path.exists( ob_list_path ):
                ob_list = load_file( ob_list_path ).tolist()
            else:
                print( "attempt to restore ob list but {0} does not exist".format( ob_list_path ) )

            if os.path.exists( t_mean_list_path ):
                t_mean_list = load_file( t_mean_list_path ).tolist()
            else:
                print( "attempt to restore t mean list but {0} does not exist".format( t_mean_list_path ) )

            if os.path.exists( t_std_list_path ):
                t_std_list = load_file( t_std_list_path ).tolist()
            else:
                print( "attempt to restore t std list  but {0} does not exist".format( t_std_list_path ) )
            
            if os.path.exists( t_logstd_list_path ):
                t_logstd_list = load_file( t_logstd_list_path ).tolist()
            else:
                print( "attempt to restore t logstd list but {0} does not exist".format( t_logstd_list_path ) )

            if os.path.exists( reward_list_path ):
                reward_list = load_file( reward_list_path ).tolist()
            else:
                print( "attempt to restore reward list but {0} does not exist".format( reward_list_path ) )

            if os.path.exists( stepped_action_list_path ):
                stepped_action_list = load_file( stepped_action_list_path ).tolist()
            else:
                print( "attempt to  restore stepped action list but {0} does not exist".format( stepped_action_list_path ) )

            num_resets = len( losses )


        # reset env
        ob = env.reset()
        ob_list.append( ob )

        if train: 

            # in this loop we accumulate enough teacher data to get us started
            while len( ob_list ) < ( LSTM_BATCH_SIZE + 1 ):
                print( "Begin Training! First Accumulate observation with teacher" )

                # accumulate observations and teacher action data
                t_mean, t_std, t_logstd  = sess.run(
                        ( teacher.pi.pd.mean, teacher.pi.pd.std, teacher.pi.pd.logstd ), 
                        feed_dict={ ob_ph: np.expand_dims( ob_list[ -1 ], axis=0 ) } )

                # accumulate teacher action data
                t_mean_list.append( np.squeeze( t_mean ) )
                t_std_list.append( np.squeeze( t_std ) )
                t_logstd_list.append( np.squeeze( t_logstd ) )


                ob, reward, new, _ = env.step( t_mean )
                reward_list.append( reward )
                stepped_action_list.append( t_mean )

                if new:
                    ob = env.reset()
                    num_resets += 1

                ob_list.append( ob )

            print( "Accumulated enough data from teacher. now train" )

            while True:

                #################### Here, we package the observation array first to to query student  ###########################
                ob_narr_list, t_mean_narr_list, t_std_narr_list, t_logstd_narr_list = [], [], [], []

                ####################################### Get teacher action ##########################################
                # Get Teacher action for the last observation
                t_mean, t_std, t_logstd  = sess.run(
                        ( teacher.pi.pd.mean, teacher.pi.pd.std, teacher.pi.pd.logstd ), 
                        feed_dict={ ob_ph: np.expand_dims( ob_list[ -1 ], axis=0 ) } )

                t_mean_list.append( np.squeeze( t_mean ) )
                t_std_list.append( np.squeeze( t_std ) )
                t_logstd_list.append( np.squeeze( t_logstd ) )
                ####################################################################################################

                for j in range( LSTM_BATCH_SIZE - 1 ):

                    # last ob does not have a corresponding action, so don't draw from the entire ob_list
                    index = randint(0, ( len(ob_list) - 1 ) - STEPS_UNROLLED )

                    # Now we make student mimic teacher
                    # These np arrays will be fed to the loss function
                    ob_narr_list.append( np.array( ob_list[index:index+STEPS_UNROLLED] ) )
                    t_mean_narr_list.append( np.array( t_mean_list[index:index+STEPS_UNROLLED] ) )
                    t_std_narr_list.append( np.array( t_std_list[index:index+STEPS_UNROLLED] ) )
                    t_logstd_narr_list.append( np.array( t_logstd_list[index:index+STEPS_UNROLLED] ) )


                # since we dont have a t_action for our last ob, 
                assert len( ob_list ) == len( t_mean_list )
                # Lastly append the last seen ob to the set of data. THIS IS NEEDED TO STEP ENV
                ob_narr_list.append( np.array( ob_list[-STEPS_UNROLLED:] ) )
                t_mean_narr_list.append( np.array( t_mean_list[-STEPS_UNROLLED:] ) )
                t_std_narr_list.append( np.array( t_std_list[-STEPS_UNROLLED:] ) )
                t_logstd_narr_list.append( np.array( t_logstd_list[-STEPS_UNROLLED:] ) )

                # make the following array STEPS_UNROLLED major i.e. dim = ( steps, batch, ob_dim )
                ob_narr = np.transpose( np.array( ob_narr_list ), (1, 0, 2) )
                t_mean_narr = np.transpose( np.array( t_mean_narr_list ), (1, 0, 2) )
                t_std_narr = np.transpose( np.array( t_std_narr_list ), (1, 0, 2) )
                t_logstd_narr = np.transpose( np.array( t_logstd_narr_list ), (1, 0, 2) )

                #################################################################################################################

                
                # Get student action for the last ovservation
                # TODO: make s_action compatible with any batch size. currently takes in LSTM_BATCH_SIZE
                s_action, curr_state  = sess.run(
                        ( s_ac, final_state_combined ), 
                        feed_dict = {
                            ob_combined_ph: ob_narr,
                            keep_prob: 1,
                            input_state_combined: curr_state } )

                ################################################################################################################

                # step with student
                ob, reward, new, _ = env.step( s_action )

                # accumulate actions are rewards
                reward_list.append( reward )
                stepped_action_list.append( np.squeeze( s_action, axis=0 ) )
                ret += reward

                # minimize loss to train student
                l,  _ = sess.run(
                            [ loss,  minimize_adam ], 
                            feed_dict = {
                                ob_combined_ph: ob_narr,
                                keep_prob: keep,
                                t_mean_combined: t_mean_narr, 
                                t_std_combined: t_std_narr, 
                                t_logstd_combined: t_logstd_narr, 
                                input_state_combined: zero_state } )
           
                # since we did not actually stepped the teacher but did generate a teacher action 
                # for this observation --> len( teacher_ac ) == len( obs )
                assert len( ob_list ) == len( t_mean_list ), "observation list length should be mapped 1:1 to teacher action"
                assert len( ob_list ) == len( reward_list ), "observation list length should be leading reward length by 1"
                assert len( ob_list ) == len( stepped_action_list ), "observation list length should be leading action length by 1"

                # minimize loss for the reward prediction model
                pred_reward, _ = sess.run(
                            [ predicted_reward, minimize_reward_l2 ], 
                            feed_dict = {
                                # upto -1 since we dont want the last observation
                                prev_ob:  np.squeeze( np.array( ob_list[ -TRAIN_REWARD_BATCH_SIZE: ] ) ), 
                                next_ac:  np.squeeze( np.array( stepped_action_list[ -TRAIN_REWARD_BATCH_SIZE: ] ) ),
                                reward_target: np.squeeze( np.array( reward_list[ -TRAIN_REWARD_BATCH_SIZE: ] ) ) } )

                pred_ret += pred_reward[-1][0]

                # record loss and return everytime environment is reset
                if ( new ):
                    ob = env.reset()
                    print ( "************** Episode {0} ****************".format( num_resets ) )
                    losses.append(l)
                    rets.append(ret)
                    pred_rets.append( pred_ret )
                    ret = 0
                    pred_ret = 0
                    save_path = saver.save(sess, lstm_trained_data_path )
                    np.save(lstm_ret_path, rets)
                    np.save(lstm_pred_ret_path, pred_rets)
                    np.save(lstm_loss_path, losses) 
                    np.save( ob_list_path, ob_list )
                    np.save( t_mean_list_path, t_mean_list )
                    np.save( t_std_list_path, t_std_list )
                    np.save( t_logstd_list_path, t_logstd_list )
                    np.save( reward_list_path, reward_list )
                    np.save( stepped_action_list_path, stepped_action_list )

                    if num_resets % 10 == 0: 

                        print( "t_mean: " )
                        print ( t_mean )
                        print( "s_action: " )
                        print( s_action )
                        print ( "KL loss: " )
                        print( l )
                        print( "actual ret" )
                        print( rets[-1] )
                        print( "predicted ret" )
                        print( pred_rets[-1] )

                    num_resets += 1

                ob_list.append( ob )

                if num_resets > 30000:
                    print( " ran {0} episodes, terminating ".format( num_resets ) )
                    save_path = saver.save(sess, lstm_trained_data_path )
                    break

        # replay mode
        else:

            ob_list = [ np.zeros([env.observation_space.shape[0] ]) for i in range(STEPS_UNROLLED - 1) ]
            ob_list.append( ob )

            for i in range(5000):

                # These np arrays will be fed to LSTM
                ob_narr_list = []
                for j in range( LSTM_BATCH_SIZE-1 ):
                    ob_narr_list.append( np.zeros([STEPS_UNROLLED, env.observation_space.shape[0]]) )
                ob_narr_list.append( np.array( ob_list[-STEPS_UNROLLED:] ) )
                ob_narr = np.transpose( ob_narr_list, (1,0,2) )

                s_action  = sess.run(
                        s_ac ,
                        feed_dict = {
                            input_state_combined: curr_state,
                            ob_combined: ob_narr } )
           

                # step with STUDENT action
                ob, reward, new, _ = env.step( s_action )
                env.render()

                if i % 100 == 0:
                    print( "s_action") 
                    print( s_action ) 

                    t_mean, t_std, t_logstd  = sess.run(
                            ( teacher.pi.pd.mean, teacher.pi.pd.std, teacher.pi.pd.logstd ), 
                            feed_dict={ ob_ph: np.expand_dims( ob_list[-1], axis=0 ) } )

                    print( "t_action" )
                    print( t_mean )
                    print( reward )

                if new:
                    ob = env.reset()
                    num_resets += 1

                    if ( num_resets % 100 ) == 0:
                        print(" Episode {0} completed!".format( num_resets ) )
                        print("********** Iteration {0} ***********".format(num_resets))

                ob_list.append( ob )

                if num_resets > 20000:
                    break

        print( "printing all trainable variables ")
        print( student_var )

    
def student_replay(klts):
    env = make_mujoco_env("Reacher-v2", 0)
    timestep = 0
    ret = 0
    episode = 0
    with tf.Session() as sess:
        student = StudentAgent(env,sess, True, klts)
        ob = env.reset()
        while episode < TOTAL_EPISODES:
            ac = student.pi.act(False, ob)
            ob, reward, new, _ = env.step(ac)
            print("********** Episode {0}, timestep {1} ***********".format(episode, timestep))
            print("reward: {0}".format(reward))
            ret += reward
            timestep += 1
            if new:
                print(ob)
                ob = env.reset()
                timestep = 0
                episode += 1
            # env.render()

######################## TRAIN ################################
#                                                             #
###############################################################
def train_student(klts):
    env = make_mujoco_env("Reacher-v2", 0)
    with tf.Session() as sess:
        # Initialize agents
        student = StudentAgent(env, sess, False, klts)
        teacher = TeacherAgent(env, sess, True)

        # This observation placeholder is for querying teacher action
       # ob_ph = U.get_placeholder( name="ob", dtype=tf.float32, 
       #       shape=[1, env.observation_space.shape[0] ] )

        ob_placeholder = U.get_placeholder( name="ob", dtype=tf.float32, shape=[ TRAINING_BATCH_SIZE ] + list(env.observation_space.shape) )
        
        # get all hidden layer variables of the student pi
        student_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope="s_pi_{0}".format("klts" if klts else "klst"))
        # print(student_var)
        teacher_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope="t_pi")

        # KL Divergence
        if klts:
            kl_div = teacher.pi.pd.kl(student.pi.pd)
        else:
            kl_div = student.pi.pd.kl(teacher.pi.pd)

        # define loss and gradient with thenos-like function
        # gradients wrt only to student variables
        lossandgrad = U.function([ob_placeholder ], [kl_div] + [U.flatgrad(kl_div, student_var)] )

        logstd = U.function([ob_placeholder], [teacher.pi.pd.logstd, student.pi.pd.logstd])
        std = U.function([ob_placeholder], [teacher.pi.pd.std, student.pi.pd.std])
        mean = U.function([ob_placeholder], [teacher.pi.pd.mean, student.pi.pd.mean])

        # initialize only student variables
        U.initialize(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                     scope="s_pi_{0}".format("klts" if klts else "klst"))
                     )

        # Adam optimiizer
        adam = MpiAdam(student_var, epsilon=1e-3)
        adam.sync()

        ob = env.reset()

        obs = []
        losses = []
        timesteps = []
        rets = []
        ret = 0
        num_resets = 0


        saver = tf.train.Saver(
          var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='s_pi_{0}'.format("klts" if klts else "klst")))

        # saver.restore(sess, "/Users/winstonww/RL/reacher_v1/student_{0}.ckpt".format("klts" if klts else "klst"))

        for timestep in range(1, TOTAL_EPISODES * TIMESTEPS_PER_EPISODE):
            # sample action
            # feed obs dict of size two 
            #  append to zeros of size [100.11], so that we can use the same
            #  model to query and train at the same time
            ob = np.expand_dims(ob,axis=0) +  np.zeros([TRAINING_BATCH_SIZE] + list( env.observation_space.shape))
            s_ac, _  = student.pi.act(False, ob)
            # print( " ob size: {0} ".format(ob.shape))
            # print( "s_ac shape" )
            # print( s_ac.shape )
            # tread along the student trajectory
            ob, reward, new, _ = env.step(s_ac)
            ret += reward
            if new:
                rets.append(ret)
                ret = 0
                ob = env.reset()
                num_resets += 1
                if num_resets > 40000: 
                    break
            # env.render()
            # print( "ob to be appended: {0}".format(ob))
            obs.append(ob)

            # compute newloss and its gradient from the two actions sampled
            # if (timestep % TRAINING_BATCH_SIZE != 0 or not timestep):
            #     continue

            # accumulate more samples before starting
            if len(obs) < TRAINING_BATCH_SIZE:
                continue

            d = Dataset(dict(ob=np.array(obs)))
            batch = d.next_batch( TRAINING_BATCH_SIZE )

            newloss, g = lossandgrad(np.squeeze(np.stack(list(batch.values()),axis=0),axis=0))
            adam.update(g,0.001)

            
            # record the following data only when reset to save time
            if new:
                losses.append(sum(newloss))
                timesteps.append(timestep)

                if num_resets % 100 == 0:
                    print("********** Episode {0} ***********".format(num_resets))
                    print("obs: \n{0}".format( np.squeeze(np.stack(list(batch.values()),axis=0),axis=0) ))
                    t_m, s_m = mean(  np.squeeze(np.stack(list(batch.values()),axis=0),axis=0) )
                    t_std, s_std = std(  np.squeeze(np.stack(list(batch.values()),axis=0),axis=0)  )
                    print("student pd std: \n{0}".format(s_std))
                    print("teacher pd std: \n{0}".format(t_std))
                    print("student pd mean: \n{0}".format(s_m))
                    print("teacher pd mean: \n{0}".format(t_m))
                    print("KL divergence: \n{0}".format(sum(newloss)))


            if timestep % 5000 == 0:
                # save results
                np.save(klts_training_loss_path if klts else klst_training_loss_path, losses) 
                np.save(klts_training_ret_path if klts else klst_training_ret_path , rets) 
                # save kl 
                save_path = saver.save(sess, "/Users/winstonww/RL/reacher_v1/student_{0}.ckpt".format("klts" if klts else "klst"))

        # save results
        np.save(klts_training_loss_path if klts else klst_training_loss_path, losses) 
        np.save(klts_training_ret_path if klts else klst_training_ret_path , rets) 

        save_path = saver.save(sess, "/Users/winstonww/RL/reacher_v1/student_{0}.ckpt".format("klts" if klts else "klst"))
    

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("-rklts" , "--replay_student_trained_with_klts", help="this mode only does student [trained with kl(teacher||student)] replay", action="store_true")
     parser.add_argument("-rklst" , "--replay_student_trained_with_klst", help="this mode only does student [trained with kl(student||teacher)]replay", action="store_true")
     parser.add_argument("-t"     , "--replay_teacher", help="this mode only does teacher replay", action="store_true")
     parser.add_argument("-klts"  , "--train_with_klts", help="training with KL(t||s)", action="store_true")
     parser.add_argument("-klst"  , "--train_with_klst", help="training with KL(s||t)", action="store_true")
     parser.add_argument("-a"     ,"--all", help="do all", action="store_true")
     parser.add_argument("-lt"     ,"--lstm_train", help="train lstm", action="store_true")
     parser.add_argument("-k"     ,"--keep_prob", help=" keep_prob on lstm ob dropout ", nargs=1, default=[1] )
     parser.add_argument("-lr"     ,"--lstm_replay", help="lstm replay", action="store_true")
     parser.add_argument("-c"     ,"--check", help="check point", action="store_true")
     parser.add_argument("-r"     ,"--restore", help="restore", action="store_true")
     parser.add_argument("-d"     ,"--date", help="date", nargs=1, default=[ datetime.datetime.now().strftime( "%Y%m%d" ) ] )
     args = parser.parse_args()


     base_path =  "{0}/reacher/data".format( str(Path.home()) )
     date_path = "{0}/{1}".format(base_path, args.date[0] )
     mlp_path = date_path + "/mlp"
     lstm_path = date_path + "/lstm"
     if not (os.path.isdir(date_path)):
         os.mkdir(date_path)
     if not (os.path.isdir(mlp_path)):
         os.mkdir(mlp_path)
     if not (os.path.isdir(lstm_path)):
         os.mkdir(lstm_path)

     klst_training_loss_path  =               "{0}/klst_KL.npy".format(mlp_path)
     klst_training_ret_path =                 "{0}/klst_ret.npy".format(mlp_path)
     teacher_ret_path =                       "{0}/teacher_ret.npy".format(mlp_path)
     lstm_trained_data_path =                 "{0}/lstm_with_keep_probability_{1}.ckpt".format( base_path, args.keep_prob[0] )
     lstm_loss_path =                         "{0}/lstm_with_keep_probability_{1}_KL.npy".format( lstm_path, args.keep_prob[0] )
     lstm_ret_path =                          "{0}/lstm_with_keep_probability_{1}_ret.npy".format( lstm_path, args.keep_prob[0] )
     lstm_pred_ret_path =                     "{0}/lstm_with_keep_probability_{1}_pred.npy".format( lstm_path, args.keep_prob[0] )
     ob_list_path =                           "{0}/lstm_with_keep_probability_{1}_ob.npy".format( lstm_path, args.keep_prob[0] )
     t_mean_list_path =                       "{0}/lstm_with_keep_probability_{1}_t_mean.npy".format( lstm_path, args.keep_prob[0] )
     t_std_list_path =                        "{0}/lstm_with_keep_probability_{1}_t_std.npy".format( lstm_path, args.keep_prob[0] )
     t_logstd_list_path =                     "{0}/lstm_with_keep_probability_{1}_t_logstd.npy".format( lstm_path, args.keep_prob[0] )
     reward_list_path =                       "{0}/lstm_with_keep_probability_{1}_reward.npy".format( lstm_path, args.keep_prob[0] )  
     stepped_action_list_path =               "{0}/lstm_with_keep_probability_{1}_stepped_action.npy".format( lstm_path, args.keep_prob[0] )

     if args.check:
         print( " checking saved variables " )
     elif args.all:
         train_student(False)
         train_student(True)
         teacher_replay()
     else:
         if args.lstm_train:
             lstm_train( True, args.keep_prob[0], lstm_trained_data_path, args.restore )
         if args.lstm_replay:
             lstm_train( False, args.keep_prob[0], lstm_trained_data_path, args.restore )
         if args.train_with_klst:
             train_student(False)
         if args.train_with_klts:
             train_student(True)
         if args.replay_student_trained_with_klts:
             student_replay(True)
         if args.replay_student_trained_with_klst:
             student_replay(False)
         if args.replay_teacher:
             teacher_replay()

     chkp.print_tensors_in_checkpoint_file( lstm_trained_data_path, tensor_name='', all_tensors=True, all_tensor_names=True)

