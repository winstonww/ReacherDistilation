import glob
import tensorflow as tf 
from baselines.common import tf_util as U
from tensorflow.python.tools import inspect_checkpoint as chkp
from baselines.common.cmd_util import make_mujoco_env
from baselines.common.distributions import make_pdtype
import numpy as np
import gym

from distilation.dataset import Dataset
from distilation.utils import load_file
from distilation.loss import kl_loss
from distilation.student_nn import student_lstm_graph
from distilation.teacher import TeacherAgent
from distilation.config import *


def train(train, restore):

    # Initialize the environment
    env = make_mujoco_env("Reacher-v2", 0)

    # new session
    sess = tf.Session()

    pdtype = make_pdtype( env.action_space )

    # initialize teacher agent
    teacher = TeacherAgent( env, sess, True, batch = 1 )

    # This observation placeholder is for querying teacher action
    ob_ph = U.get_placeholder( name="ob", dtype=tf.float32, 
          shape=[ 1, env.observation_space.shape[0] ] )

    with tf.variable_scope("LSTM" ):

        # different from ob_ph, this tf placeholder holds a batch of observations for lstm training
        ob_batch_ph = tf.placeholder(name="ob_batch_ph", dtype=tf.float32, 
                          shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, OBSPACE_SHAPE ] )

        prev_pdflat_batch_ph = tf.placeholder( name="prev_pdflat_batch_ph", dtype=tf.float32, 
                          shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE ] )

        #prev_rew_batch_ph = tf.placeholder( name="prev_rew_batch_ph", dtype=tf.float32, 
        #                  shape=[ STEPS_UNROLLED, MLP_BATCH_SIZE, 1 ] )

        keep_prob_ph = tf.placeholder(name="keep_prob_ph",dtype=tf.float32, shape=[] ) 


        # ou#ter dim is 2 because of c_state and m state
        initial_state_batch_ph = tf.placeholder( shape=[ 2, LSTM_BATCH_SIZE, NUM_UNITS ], dtype=tf.float32 )

        # lstm graph; shape of s_pdflat_batch:[STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE]
        s_pdflat_batch, final_state_batch  = student_lstm_graph( ob_batch_ph, keep_prob_ph, prev_pdflat_batch_ph, initial_state_batch_ph  )

        t_pdflat_batch_ph = tf.placeholder( 
                name="t_pdflat_batch_ph", shape=[ STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE ], dtype=tf.float32 )

        # get student action wrt last observation
        # beginning at last index (i.e. STEPS_UNROLLED-1 ), sample down 1 element in the first (outer)  dimension
        # , and all elements in the inner dimensions
        s_pdflat_slice = tf.slice( s_pdflat_batch, [ (STEPS_UNROLLED-1), (LSTM_BATCH_SIZE-1),  0 ], [ 1, 1, -1 ] )

        # for stepping
        s_action = pdtype.pdfromflat(s_pdflat_slice).mean

    # get a collection of students within the 'LSTM' scope for optimization
    student_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="LSTM" )

    loss = kl_loss(s_pdflat_batch, t_pdflat_batch_ph, pdtype)

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

    # saver for restoring/saving depending on whether or not to train
    saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LSTM') )

    train_writer = tf.summary.FileWriter( "/home/winstonww/reacher/data/viz/1" )
    train_writer.add_graph(sess.graph)

    # state variables for lstm
    zero_state_batch = np.zeros( [ 2, LSTM_BATCH_SIZE, NUM_UNITS ] )
    curr_state_batch = zero_state_batch

    with sess:

        # run initializer for adam optimizer
        sess.run( tf.variables_initializer( adam.variables() ) )

        # run initializer for lstm variables
        if not restore:
            sess.run(init)
        elif glob.glob( lstm_trained_data_path + "*" ):
            saver.restore( sess, lstm_trained_data_path )
        else:
            print( "attempt to restore trained data but {0} does not exist".format( lstm_trained_data_path ) )

        dataset = Dataset(dir_path=dataset_path)
        # reset env
        ob = env.reset()

        reward = 0

        # in this loop we accumulate enough teacher data to get us started
        print( "Begin Training! First Accumulate observation with teacher" )

        while dataset.num_episodes() <= LSTM_BATCH_SIZE*2:

            # accumulate observations and teacher action data
            t_mean, t_pdflat  = sess.run(
                    ( teacher.pi.pd.mean, teacher.pi.pd.flat ),
                    feed_dict={ ob_ph: np.expand_dims(ob, axis=0 ) } )

            dataset.write(
                ob=ob,
                reward=reward,
                t_pdflat=t_pdflat,
                s_pdflat=np.zeros([PDFLAT_SHAPE]),
                stepped_with='t')


            ob, reward, new, _ = env.step( t_mean )

            if new:
                ob = env.reset()
                dataset.flush()

        print( "Accumulated sufficient data points from teacher. now train" )

        while True:

            # train using tensorflow's truncated propagation
            total_loss = 0 
            for (ob_batch_array, t_pdflat_batch_array, prev_pdflat_batch_array, prev_rew_batch_array ) in dataset.training_batches():
                # minimize loss to train student
                l,  _ = sess.run(
                        [ loss,  minimize_adam ], 
                        feed_dict = {
                            keep_prob_ph: KEEP_PROB,
                            ob_batch_ph: ob_batch_array,
                            # TODO: revert this back
                            prev_pdflat_batch_ph: prev_pdflat_batch_array,
                            #prev_rew_batch_ph: prev_rew_batch_array,
                            #prev_pdflat_batch_ph: ob_batch_array,
                            #prev_pdflat_batch_ph: np.random.rand(STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE),
                            #prev_pdflat_batch_ph: np.zeros([STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE]),
                            t_pdflat_batch_ph: t_pdflat_batch_array,
                            initial_state_batch_ph: zero_state_batch } )
                total_loss += l

            # Get Teacher action for the last observation
            t_pdflat  = sess.run(
                    ( teacher.pi.pd.flat ), 
                    feed_dict={ ob_ph: np.expand_dims( ob, axis=0 ) } )


            ob_batch_array, prev_pdflat_batch_array, prev_rew_batch_array = dataset.test_batch(ob)
                
            # Get student action for the last ovservation
            s_ac, s_pdflat, curr_state_batch  = sess.run(
                    ( s_action, s_pdflat_slice, final_state_batch ), 
                    feed_dict = {
                        keep_prob_ph: 1,
                        ob_batch_ph: ob_batch_array,
                        #TODO: revert this back
                        prev_pdflat_batch_ph: prev_pdflat_batch_array,
                        #prev_rew_batch_ph: prev_rew_batch_array,
                        #prev_pdflat_batch_ph: ob_batch_array,
                        #prev_pdflat_batch_ph: np.random.rand(STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE),
                        #prev_pdflat_batch_ph: np.zeros([STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE]),
                        initial_state_batch_ph: curr_state_batch } )

            dataset.write(
                ob=ob,
                reward=reward,
                t_pdflat=t_pdflat,
                s_pdflat=s_pdflat,
                stepped_with='s')
                        
            # step with student
            ob, reward, new, _ = env.step( s_ac )

            if new:
                print ( "************** Episode {0} ****************".format(dataset.num_episodes()) )
                ob = env.reset()
                print("recent loss: %f " % total_loss )
                dataset.flush()
                save_path = saver.save(sess, lstm_trained_data_path )
                if dataset.num_episodes() % MAX_CAPACITY == 0: dataset.dump()
                if dataset.num_episodes() == TOTAL_EPISODES: break
