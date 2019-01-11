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
TOTAL_EPISODES = 2
TIMESTEPS_PER_EPISODE = 50
SLIDING_WINDOW = 200

# # number of steps to unroll 
# STEPS_UNROLLED = 20
# LSTM_BATCH_SIZE=100
# NUM_UNITS=100
# 
# preward prediction training
EPISODE_STEPS = 50
GAMMA = 0.99

# number of steps to unroll 
STEPS_UNROLLED = 2
LSTM_BATCH_SIZE=2
NUM_UNITS=1

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
# The method takes in input_combined
# and returns stacked action mean, logstd and std of the student, and the final state.
def lstm_graph( input_combined, hidden_combined, env ):

    # parse action distribution mean, logstd
    # get action space type
    pdtype = make_pdtype( env.action_space )

    # new cell
    cell = tf.contrib.rnn.LSTMCell( num_units=NUM_UNITS, name="unique_lstm_cell" )

    # Initailize state with zero of batch size 1 and type float32
    #c_state, m_state = tf.split(hidden_combined, [1, 1], 0 ) 
    c_state, m_state = hidden_combined[0,:,:], hidden_combined[1,:,:]

    state = tf.tuple( [ c_state, m_state ], name="cm_state" )

    reward_list, s_mean_list, s_std_list, s_logstd_list = [], [], [], []

    for i in range(STEPS_UNROLLED):

        # if i > 0: tf.get_variable_scope().reuse_variables()
        # normalize observation vector with rms
        #rms = RunningMeanStd( shape=env.observation_space.shape )
        # only first step; the rest will need all (all batch, all observation space dim )
        #obz = tf.clip_by_value( (input_combined[i, :, :] - rms.mean) / rms.std, -5.0, 5.0 )
        output, next_state = cell( input_combined[ i, :, : ], state )

        with tf.name_scope( "step%d"%(i+1) ) as scope:

            dense = tf.nn.tanh( tf.layers.dense(output, 128, name='lstm_dense' ), name='lstm_dense_tanh' )

            reward = tf.nn.tanh( tf.layers.dense( dense, 64, name='reward1_dense' ),  name='reward1_tanh' )
            # reward = tf.nn.tanh( tf.layers.dense( reward, 32 ), 'reward2' )
            # reward = tf.nn.tanh( tf.layers.dense( reward, 64 ), 'reward3' )
            reward = tf.layers.dense( reward, 1, name="reward_out" )

            lstm_action = tf.nn.tanh( tf.layers.dense( dense, 64, name='action_dense' ), name="action_tanh" )


            with tf.name_scope( "from_flat_scope" ) as scope:
                # feed the output of lstm to a final FC layer
                # this 'flat' vector will be split into the mean and std of a pd
                pdparam = tf.layers.dense( lstm_action, pdtype.param_shape()[0], name=scope )
                pd = pdtype.pdfromflat( pdparam )

        s_mean_list.append( pd.mean )
        s_std_list.append( pd.std )
        s_logstd_list.append( pd.logstd )
        reward_list.append( reward )

    # stack the outputs at each cell together so that we can conveniently compute loss and etc
    with tf.name_scope( "mean_stack" ) as scope:
        s_mean_combined = tf.stack( s_mean_list, name=scope )
    with tf.name_scope( "std_stack" ) as scope:
        s_std_combined = tf.stack( s_std_list, name=scope )
    with tf.name_scope( "log_stack" ) as scope:
        s_logstd_combined = tf.stack( s_logstd_list, name=scope )
    with tf.name_scope( "final_state" ) as scope:
        final_state_combined = tf.identity( state, name=scope )
    with tf.name_scope( "reward_stack" ) as scope:
        reward_combined = tf.squeeze( tf.stack( reward_list ), name=scope )

    return s_mean_combined, s_std_combined, s_logstd_combined, final_state_combined, reward_combined


# This method compute this total loss 
def lstm_loss( s_mean_combined, s_std_combined, s_logstd_combined, 
        t_mean_combined, t_std_combined, t_logstd_combined, env ):
    
    return tf.reduce_sum(t_logstd_combined - s_logstd_combined + 
            (tf.square(s_std_combined) + tf.square(s_mean_combined - t_mean_combined)) / (2.0 * tf.square(t_std_combined)) - 0.5, axis=[2,1,0], name="lstm_kl_loss" )


# This method generates the training set
def generate_training_set( ob_list, t_mean_list, t_std_list, t_logstd_list, stepped_action_list, reward_list, num_episodes_completed ):

    # length of reward of stepped_action list is shorter by 1
    assert len( ob_list ) == len( t_mean_list ) == len( t_std_list ) == len( t_logstd_list ) == ( len( stepped_action_list ) + 1 ) == ( len( reward_list ) + 1 )

    reward_narr_list, stepped_action_narr_list, ob_narr_list, t_mean_narr_list, t_std_narr_list, t_logstd_narr_list = [], [], [], [], [], []

    for k in range( LSTM_BATCH_SIZE ):

        # last ob does not have a corresponding action, so don't draw from the entire ob_list
        if num_episodes_completed - 1 < 0:
            index = 0
        else:
            index = randint( 0, ( num_episodes_completed - 1 ) ) * EPISODE_STEPS

        index += randint( 1, EPISODE_STEPS - STEPS_UNROLLED - 1 ) 


        ob_narr_list.append( np.array( ob_list[ index:index+STEPS_UNROLLED ] ) )
        # here we use the previous action- the action we stepped to generate the above ob to compute q(s,a); hence index-1
        stepped_action_narr_list.append( np.array( stepped_action_list[index-1:index-1+STEPS_UNROLLED] ) )
        reward_narr_list.append( np.array( reward_list[ index:index+STEPS_UNROLLED ]) )
        t_mean_narr_list.append( np.array( t_mean_list[ index:index+STEPS_UNROLLED ] ) )
        t_std_narr_list.append( np.array( t_std_list[ index:index+STEPS_UNROLLED ] ) )
        t_logstd_narr_list.append( np.array( t_logstd_list[ index:index+STEPS_UNROLLED ] ) )

    # make the following array STEPS_UNROLLED major i.e. dim = ( steps, batch, ob_dim )
    ob_narr       = np.transpose( np.array( ob_narr_list ), ( 1, 0, 2 ) )
    reward_narr   = np.transpose( np.array( reward_narr_list ), ( 1, 0 ) )
    action_narr   = np.transpose( np.array( stepped_action_narr_list ), ( 1, 0, 2 ) )
    t_mean_narr   = np.transpose( np.array( t_mean_narr_list ), ( 1, 0, 2 ) )
    t_std_narr    = np.transpose( np.array( t_std_narr_list ), ( 1, 0, 2 ) )
    t_logstd_narr = np.transpose( np.array( t_logstd_narr_list ), ( 1, 0, 2 ) )

    return ob_narr, t_mean_narr, t_std_narr, t_logstd_narr, action_narr, reward_narr


def generate_test_set( ob, stepped_action, env ):

    stepped_action_narr_list, ob_narr_list = [], []

    for k in range( LSTM_BATCH_SIZE - 1):
        # fill with zeros
        ob_narr_list.append( np.zeros( ( STEPS_UNROLLED, env.observation_space.shape[0] ) ) )
        stepped_action_narr_list.append( np.zeros( ( STEPS_UNROLLED, env.action_space.shape[0] ) ) )

    # Lastly append the last seen ob to the set of data. THIS IS NEEDED TO STEP ENV
    ob_narr_list.append( np.array( ob ) )
    # Here since the length of stepped_action_list is one less than ob
    stepped_action_narr_list.append( stepped_action )

    # make the following array STEPS_UNROLLED major i.e. dim = ( steps, batch, ob_dim )
    ob_narr       = np.transpose( np.array( ob_narr_list ), ( 1, 0, 2 ) )
    action_narr   = np.transpose( np.array( stepped_action_narr_list ), ( 1, 0, 2 ) )

    return ob_narr, action_narr


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

    with tf.variable_scope("LSTM", reuse=tf.AUTO_REUSE ):

        # this tf placeholder holds a batch of observations for lstm training
        ob_combined_ph = tf.placeholder(name="ob_combined_ph", dtype=tf.float32, 
                          shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.observation_space.shape[0] ] )


        action_combined_ph = tf.placeholder( name="action_combined_ph", dtype=tf.float32, 
                          shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.action_space.shape[0] ] )

        # keep probability
        keep_prob = tf.placeholder(name="keep_prob",dtype=tf.float32, shape=[] ) 

        # combine the observation and previous action to create a flat representing the input, and apply dropout
        ob_combined_dropout = tf.nn.dropout( ob_combined_ph, keep_prob )

        input_combined = tf.concat( [ ob_combined_dropout, action_combined_ph ], -1 )

        # input state placeholder, outer dim is 2 because of c_state and m state
        hidden_combined = tf.placeholder( shape=[ 2, LSTM_BATCH_SIZE, NUM_UNITS ], dtype=tf.float32 )

        # state variables for lstm
        zero_state = np.zeros( [ 2, LSTM_BATCH_SIZE, NUM_UNITS ] )
        curr_state = zero_state

        # lstm graph; shape of s_mean_combined is still [STEPS_UNROLLED, LSTM_BATCH_SIZE, 2]
        s_mean_combined, s_std_combined, s_logstd_combined, final_state_combined, reward_combined  = lstm_graph( input_combined, hidden_combined, env )

        # placeholder for teacher action paramters, shape is [5, 1, 2]
        t_mean_combined = tf.placeholder( 
                name="t_mean_combined",     shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.action_space.shape[0] ], dtype=tf.float32 )

        t_std_combined = tf.placeholder( 
                name="t_std_combined",       shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.action_space.shape[0] ], dtype=tf.float32 )

        t_logstd_combined = tf.placeholder( 
                name="t_logstd_combined", shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, env.action_space.shape[0] ], dtype=tf.float32 )

        reward_target = tf.placeholder( name="reward_target", shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE ], dtype=tf.float32 )
        
        # get the action wrt last observation
        # beginning at last index (i.e. STEPS_UNROLLED-1 ), sample down 1 element in the first (outer)  dimension
        # , and all elements in the inner dimensions
        s_ac = tf.slice( s_mean_combined, [ ( STEPS_UNROLLED - 1 ), ( LSTM_BATCH_SIZE - 1 ),  0 ], [ 1, 1, -1 ] )

        # get kl loss 
        with tf.name_scope( "loss_scope" ):
            ll = lstm_loss( s_mean_combined, s_std_combined, s_logstd_combined, t_mean_combined, t_std_combined, t_logstd_combined, env )

            loss = tf.add( ll,  tf.reduce_sum( tf.square( reward_combined - reward_target ) ), name="total_loss" )

    # get a collection of students within the 'LSTM' scope for optimization
    student_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="LSTM" )

    ################################################################################################################


    with tf.name_scope( "adam" ):
        # adam optimizer for minimize kl loss; learning rate is fixed here
        adam = tf.train.AdamOptimizer( learning_rate=1e-3,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-8)
        
        minimize_adam = adam.minimize( loss, var_list=student_var )

    # initializer; to be placed at the very end
    init = tf.variables_initializer( 
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="LSTM") )

    # keep track of env resets i.e. num of episodes
    num_episodes_completed = 0

    # saver for restoring/saving depending on whether or not to train
    saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM') )

    # these lists are for training the policy
    ob_list, t_mean_list, t_std_list, t_logstd_list, reward_list, stepped_action_list  = [], [], [], [], [], []

    # save loss kl and rets
    losses, rets = [], []

    ret = 0

    train_writer = tf.summary.FileWriter( "/home/winstonww/reacher/data/viz/1" )
    train_writer.add_graph( sess.graph)

    with sess:

        if not train:
            sess.run(init)
            saver.restore( sess, lstm_trained_data_path )
            ob = env.reset()
            ob_list = np.zeros( ( STEPS_UNROLLED, env.observation_space.shape[0] ) )
            action_list = np.zeros( ( STEPS_UNROLLED, env.action_space.shape[0] ) )
            
            ob_narr, action_narr = generate_test_set( ob_list, action_list, env )

            while True:

                # Get student action for the last ovservation
                # TODO: make s_action compatible with any batch size. currently takes in LSTM_BATCH_SIZE
                s_action, curr_state  = sess.run(
                        ( s_ac, final_state_combined ), 
                        feed_dict = {
                            ob_combined_ph: ob_narr,
                            action_combined_ph: action_narr,
                            keep_prob: 1,
                            hidden_combined: curr_state } )

                ob, reward, new, _ = env.step( s_action )
                action_list[-1,:] = s_action
                ob_list[-1,:] = ob

                if new:
                    print( "New Episode!" )
                    ob = env.reset()

            return

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
                print( "attempt to restore stepped action list but {0} does not exist".format( stepped_action_list_path ) )

            num_episodes_completed = len( losses )

        # reset env
        ob = env.reset()
        ob_list.append( ob )

        if train: 

            # in this loop we accumulate enough teacher data to get us started
            print( "Begin Training! First Accumulate observation with teacher" )

            while len( ob_list ) < ( LSTM_BATCH_SIZE + 1 ) or len( ob_list ) < EPISODE_STEPS:

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
                stepped_action_list.append( np.squeeze( t_mean ) )

                if new:
                    ob = env.reset()
                    num_episodes_completed += 1

                ob_list.append( ob )

            print( "Accumulated {0} data points from teacher. now train".format( len( stepped_action_list) ) )

            assert ( len( ob_list ) - 1 ) == len( t_mean_list ) == len( t_std_list ) == len( t_logstd_list ) == ( len( stepped_action_list ) ) == ( len( reward_list ) )

            timestep = 0

            while True:

                ####################################### Get teacher action ##########################################

                # Get Teacher action for the last observation
                t_mean, t_std, t_logstd  = sess.run(
                        ( teacher.pi.pd.mean, teacher.pi.pd.std, teacher.pi.pd.logstd ), 
                        feed_dict={ ob_ph: np.expand_dims( ob_list[ -1 ], axis=0 ) } )

                t_mean_list.append( np.squeeze( t_mean ) )
                t_std_list.append( np.squeeze( t_std ) )
                t_logstd_list.append( np.squeeze( t_logstd ) )

                #################### Here, we package the observation array first to to query student  ###########################

                ob_narr, t_mean_narr, t_std_narr, t_logstd_narr, action_narr, reward_narr = \
                    generate_training_set( ob_list, t_mean_list, t_std_list, t_logstd_list, stepped_action_list, reward_list, num_episodes_completed )

                # minimize loss to train student
                l,  _ = sess.run(
                            [ loss,  minimize_adam ], 
                            feed_dict = {
                                ob_combined_ph: ob_narr,
                                action_combined_ph: action_narr,
                                keep_prob: keep,
                                t_mean_combined: t_mean_narr, 
                                t_std_combined: t_std_narr, 
                                t_logstd_combined: t_logstd_narr, 
                                reward_target: reward_narr,
                                hidden_combined: zero_state } )

                ob_narr, action_narr = generate_test_set( ob_list[-STEPS_UNROLLED:], stepped_action_list[-STEPS_UNROLLED:], env )
                #################################################################################################################
                
                # Get student action for the last ovservation
                # TODO: make s_action compatible with any batch size. currently takes in LSTM_BATCH_SIZE
                s_action, curr_state  = sess.run(
                        ( s_ac, final_state_combined ), 
                        feed_dict = {
                            ob_combined_ph: ob_narr,
                            action_combined_ph: action_narr,
                            keep_prob: 1,
                            hidden_combined: curr_state } )

                ################################################################################################################

                # step with student
                ob, reward, new, _ = env.step( s_action )
                ret += pow( GAMMA, timestep ) * reward
                timestep += 1

                # accumulate actions are rewards
                reward_list.append( reward )
                stepped_action_list.append( np.squeeze( s_action ) )

           
                # since we did not actually stepped the teacher but did generate a teacher action 
                # for this observation --> len( teacher_ac ) == len( obs )
                assert len( ob_list ) == len( t_mean_list ) == len( reward_list ) == len( stepped_action_list )

                # record loss and return everytime environment is reset
                if ( new ):
                    ob = env.reset()
                    print ( "************** Episode {0} ****************".format( num_episodes_completed ) )
                    losses.append(l)
                    rets.append(ret)
                    timestep = ret = 0
                    save_path = saver.save(sess, lstm_trained_data_path )
                    np.save(lstm_ret_path, rets )
                    np.save(lstm_loss_path, losses )
                    np.save( ob_list_path, ob_list )
                    np.save( t_mean_list_path, t_mean_list )
                    np.save( t_std_list_path, t_std_list )
                    np.save( t_logstd_list_path, t_logstd_list )
                    np.save( reward_list_path, reward_list )
                    np.save( stepped_action_list_path, stepped_action_list )

                    if num_episodes_completed % 10 == 0: 

                        print( "t_mean: " )
                        print ( t_mean )
                        print( "s_action: " )
                        print( s_action )
                        print ( "Total loss: " )
                        print( l )
                        print( "actual return:" )
                        print( rets[-1] )

                    num_episodes_completed += 1

                ob_list.append( ob )

                if num_episodes_completed > TOTAL_EPISODES:
                    print( " ran {0} episodes, terminating ".format( num_episodes_completed ) )
                    save_path = saver.save(sess, lstm_trained_data_path )
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
        num_episodes_completed = 0


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
            # tread along the student trajectory
            ob, reward, new, _ = env.step(s_ac)
            ret += reward
            if new:
                rets.append(ret)
                ret = 0
                ob = env.reset()
                num_episodes_completed += 1
                if num_episodes_completed > 40000: 
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

                if num_episodes_completed % 100 == 0:
                    print("********** Episode {0} ***********".format(num_episodes_completed))
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

