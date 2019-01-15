#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import os
import datetime
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt

####################### GLOBAL DEFINES ########################
#                                                             #
###############################################################

STARTING_EPISODES = 0
TOTAL_EPISODES = 500
SLIDING_WINDOW = 2
X_LIMIT = TOTAL_EPISODES / SLIDING_WINDOW
TIMESTEPS_PER_EPISODE = 50

def load_file(path):
    arr = []
    if not os.path.exists(path):
        print("Source npy does not exists @ {0}".format(path))
        # warnings.warn("Source npy does not exists")
    else:
        arr = np.load(path)
    return arr


# plot training loss teacher saved previously
def plot( plot_all, plot_teacher, plot_klts, plot_klst, plot_lstm, plot_dropout ):

    num_fig = 1
    # klts 
    if plot_all or plot_klts:
        klts_training_losses = load_file(klts_training_loss_path)
        klts_training_losses = klts_training_losses[STARTING_EPISODES:TOTAL_EPISODES]
        loss_len = len( klts_training_losses )
        klts_training_losses = [ sum(klts_training_losses[i:i+SLIDING_WINDOW]) / (SLIDING_WINDOW)
                for i in range(0, loss_len, SLIDING_WINDOW) ]

        klts_training_ret = load_file(klts_training_ret_path)
        # only for the first 6000 to compare with klts
        klts_training_ret = klts_training_ret[STARTING_EPISODES:TOTAL_EPISODES]

        ret_len = len( klts_training_ret )

        klts_training_ret = [ sum(klts_training_ret[i:i+SLIDING_WINDOW]) / (TIMESTEPS_PER_EPISODE * SLIDING_WINDOW)
                for i in range(0, ret_len, SLIDING_WINDOW) ]

        plt.figure(num_fig)
        num_fig += 1
        plt.xlabel("Episode (every {0}'th)".format( SLIDING_WINDOW ))
        plt.ylabel("KL divergence")
        plt.plot(range(len(klts_training_losses)), klts_training_losses)
        plt.title("Average KL divergence MLP of every {0} epsisodes for {1} episodes ".format( SLIDING_WINDOW, len( klts_training_losses ) * SLIDING_WINDOW ))
        plt.axis( [ 0, X_LIMIT, -0.5, 400 ] )
        plt.savefig("{0}/klts.png".format(mlp_path))
        print( 'klts_loss')
        print(klts_training_losses)

        fig = plt.figure(num_fig)
        num_fig += 1
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Episode (every {0}'th)".format( SLIDING_WINDOW ))
        ax.set_ylabel("Average reward")
        print( 'klts_ret')
        # ax.set_yscale('log')
        ax.plot(klts_training_ret)
        ax.set_title("Average student reward of every {0} episodes ({1} in total)".format(SLIDING_WINDOW, ret_len))
        plt.savefig("{0}/student_klts_rewards.png".format(mlp_path))

    # klst
    if plot_all or plot_klst:
        klst_training_ret = load_file(klst_training_ret_path)
        klst_training_loss = load_file(klst_training_loss_path)
        klst_training_ret = [ sum(klst_training_ret[i:i+SLIDING_WINDOW]) / ( TIMESTEPS_PER_EPISODE * SLIDING_WINDOW )
                for i in range(0,TOTAL_EPISODES, SLIDING_WINDOW) ]
        plt.figure(num_fig)
        num_fig += 1
        plt.xlabel("timestep")
        plt.ylabel("KL divergence")
        plt.plot(range(len(klst_training_loss)), klst_training_loss)
        plt.title("KL divergence (student||teacher)")
        plt.savefig("{0}/klst.png".format(date_path))

        fig = plt.figure(num_fig)
        num_fig += 1
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Episodes (hundreds)")
        ax.set_ylabel("Average reward")
        # ax.set_yscale('log')
        ax.plot(klst_training_ret)
        ax.set_title(" Average student reward of every {0} episodes ({1} in total)".format(SLIDING_WINDOW, TOTAL_EPISODES))
        plt.savefig("{0}/student_klst_rewards.png".format(date_path))


    if plot_all or plot_teacher:
        # history of env reward (teacher rollout)
        teacher_ret = load_file(teacher_ret_path)
        teacher_ret = teacher_ret[STARTING_EPISODES:TOTAL_EPISODES]
        teacher_ret       = [ sum(teacher_ret[i:i+SLIDING_WINDOW]) / (TIMESTEPS_PER_EPISODE * SLIDING_WINDOW)
                for i in range(0, len(teacher_ret), SLIDING_WINDOW) ]
        fig = plt.figure(num_fig)
        num_fig += 1
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Episode #")
        ax.set_ylabel("Average reward")
        # ax.set_yscale('log')
        ax.plot(range(len(teacher_ret)), teacher_ret)
        ax.set_title("Average teacher reward  of every {0} episodes ({1} in total)".format(SLIDING_WINDOW, TOTAL_EPISODES))
        plt.savefig("{0}/teacher_rewards.png".format(teacher_path))

    
    if plot_all or plot_lstm:
        lstm_training_losses = load_file( lstm_loss_path )
        lstm_training_losses = lstm_training_losses[STARTING_EPISODES:TOTAL_EPISODES]
        loss_len = len( lstm_training_losses )
        lstm_training_losses = [ sum(lstm_training_losses[i:i+SLIDING_WINDOW]) / ( SLIDING_WINDOW)
                for i in range(0, loss_len, SLIDING_WINDOW) ]

        lstm_training_ret = load_file( lstm_ret_path )
        # only for the first 6000 to compare with klts
        lstm_training_ret = lstm_training_ret[STARTING_EPISODES:TOTAL_EPISODES]
        ret_len = len( lstm_training_ret )
        lstm_training_ret = [ sum(lstm_training_ret[i:i+SLIDING_WINDOW]) / (TIMESTEPS_PER_EPISODE * SLIDING_WINDOW)
                for i in range(0, ret_len, SLIDING_WINDOW) ]
        print( 'lstm_ret')

        plt.figure(num_fig)
        num_fig += 1
        plt.xlabel("Episode (every {0}'th)".format( SLIDING_WINDOW ))
        plt.ylabel("KL divergence")
        plt.axis( [ 0, X_LIMIT, -0.5, 400 ] )
        plt.plot(range(len(lstm_training_losses)), lstm_training_losses)
        plt.title("Average KL divergence LSTM of every {0} epsisodes for {1} episodes ".format( SLIDING_WINDOW, len( lstm_training_losses ) * SLIDING_WINDOW ))
        plt.savefig("{0}/lstm_kl.png".format(lstm_path))
        print( 'lstm_loss')

        fig = plt.figure(num_fig)
        num_fig += 1
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Episode (every {0}'th) beginning at episode {1} to episode {2}".format( SLIDING_WINDOW, STARTING_EPISODES, TOTAL_EPISODES ))
        ax.set_ylabel("Average reward")
        # ax.set_yscale('log')
        ax.plot(lstm_training_ret)
        ax.set_title(" Average student reward of every {0} episodes ({1} in total)".format(SLIDING_WINDOW, len( lstm_training_ret ) * SLIDING_WINDOW))
        plt.savefig("{0}/lstm_training_ret.png".format(lstm_path))
        print( lstm_path )

    if plot_all or plot_dropout:
        lstm_dropout_losses = load_file( lstm_dropout_loss_path )
        lstm_dropout_losses = lstm_dropout_losses[STARTING_EPISODES:TOTAL_EPISODES]
        loss_len = len( lstm_dropout_losses )
        lstm_dropout_losses = [ sum(lstm_dropout_losses[i:i+SLIDING_WINDOW]) / ( SLIDING_WINDOW)
                for i in range(0, loss_len, SLIDING_WINDOW) ]

        lstm_dropout_ret = load_file( lstm_dropout_ret_path )
        lstm_dropout_ret = lstm_dropout_ret[STARTING_EPISODES:TOTAL_EPISODES]
        ret_len = len( lstm_dropout_ret )

        lstm_dropout_ret = [ sum(lstm_dropout_ret[i:i+SLIDING_WINDOW]) / (TIMESTEPS_PER_EPISODE * SLIDING_WINDOW)
                for i in range(0, ret_len, SLIDING_WINDOW) ]
        print( 'lstm_ret')

        plt.figure(num_fig)
        num_fig += 1
        plt.xlabel("Episode (every {0}'th)".format( SLIDING_WINDOW ))
        plt.ylabel("KL divergence")
        plt.axis( [ 0, X_LIMIT, -0.5, 400 ] )
        plt.plot(range(len(lstm_dropout_losses)), lstm_dropout_losses)
        plt.title("Average KL divergence LSTM of every {0} epsisodes for {1} episodes ".format( SLIDING_WINDOW, len( lstm_dropout_losses ) * SLIDING_WINDOW ))
        plt.savefig("{0}/lstm_dropout_kl.png".format(lstm_path))
        print( 'lstm_dropout_loss')

        fig = plt.figure(num_fig)
        num_fig += 1
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Episode (every {0}'th) beginning at episode {1} to episode {2}".format( SLIDING_WINDOW, STARTING_EPISODES, TOTAL_EPISODES ))
        ax.set_ylabel("Average reward")
        # ax.set_yscale('log')
        ax.plot(lstm_dropout_ret)
        ax.set_title(" Average student reward of every {0} episodes ({1} in total)".format(SLIDING_WINDOW, ( TOTAL_EPISODES - STARTING_EPISODES ) * SLIDING_WINDOW))
        plt.savefig("{0}/lstm_dropout_ret.png".format(lstm_path))
        print( lstm_path )

        fig = plt.figure( num_fig )
        num_fig += 1
        ax = fig.add_subplot( 1, 1, 1 )
        ax.set_xlabel("Episode (every {0}'th) beginning at episode {1} to episode {2}".format( SLIDING_WINDOW, STARTING_EPISODES, TOTAL_EPISODES ))
        ax.set_ylabel("Average reward")
        ax.set_title(" Average student reward of every {0} episodes ({1} in total)".format(SLIDING_WINDOW, ( TOTAL_EPISODES - STARTING_EPISODES ) * SLIDING_WINDOW))
        ax.plot(lstm_dropout_ret, label="lstm_with_dropout")
        ax.plot(lstm_training_ret, label="lstm")
        ax.plot(klts_training_ret, label="klts")
        ax.plot(teacher_ret, label="teacher" )
        ax.legend()
        plt.savefig("{0}/ret_lstm_klts.png".format(lstm_path))

        fig = plt.figure( num_fig )
        num_fig += 1
        ax = fig.add_subplot( 1, 1, 1 )
        ax.set_xlabel("Episode (every {0}'th) beginning at episode {1} to episode {2}".format( SLIDING_WINDOW, STARTING_EPISODES, TOTAL_EPISODES ))
        ax.set_ylabel("KL Divergence")
        ax.set_title(" KL Divergence of every {0} episodes ({1} in total)".format(SLIDING_WINDOW, ( TOTAL_EPISODES - STARTING_EPISODES )  * SLIDING_WINDOW))
        ax.plot(lstm_dropout_losses, label="lstm_with_dropout")
        ax.plot(lstm_training_losses, label="lstm")
        ax.legend()
        plt.savefig("{0}/loss_lstm_klts.png".format(lstm_path))



    if plot_all:
        fig = plt.figure(num_fig)
        num_fig += 1
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Average reward")
        # ax.set_yscale('log')
        line1 = ax.plot(range(SLIDING_WINDOW, (len(teacher_ret)+1)*SLIDING_WINDOW,       SLIDING_WINDOW), 
                teacher_ret, label="teacher reward")
        line2 = ax.plot(range(SLIDING_WINDOW, (len(klst_training_ret)+1)*SLIDING_WINDOW, SLIDING_WINDOW), 
                klst_training_ret, label="student_klst reward")
        line3 = ax.plot(range(SLIDING_WINDOW, (len(klts_training_ret)+1)*SLIDING_WINDOW, SLIDING_WINDOW), 
                klts_training_ret, label="student klts reward")
        ax.legend()
        ax.set_title(" Average student and teacher reward of every {0} episodes ({1} in total)".format(SLIDING_WINDOW, TOTAL_EPISODES))
        plt.savefig("{0}/combined_rewards_sw_{1}_tot_{2}.png".format(date_path, SLIDING_WINDOW, TOTAL_EPISODES))

    return

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("-t"     , "--plot_teacher", help="this mode only does teacher replay", action="store_true")
     parser.add_argument("-klts"  , "--plot_klts", help="plot (t||s)", action="store_true")
     parser.add_argument("-klst"  , "--plot_klst", help="plot KL(s||t)", action="store_true")
     parser.add_argument("-lstm"  , "--plot_lstm", help="plot lstm", action="store_true")
     parser.add_argument("-o"     , "--plot_dropout", help="plot dropout", action="store_true")
     parser.add_argument("-a"     ,"--plot_all", help="do all", action="store_true")
     parser.add_argument("-d"     ,"--date", help="date dir containing data ", nargs=1, 
             default=[ datetime.datetime.now().strftime( "%Y%m%d" ) ] )
     args = parser.parse_args()


     base_path =  "{0}/reacher/data".format( str( Path.home() ) )
     date = args.date[0]
     date_path = "{0}/{1}".format(str(Path.home()), date )
     mlp_path = date_path + "/mlp"
     lstm_path = date_path + "/lstm"
     teacher_path = date_path + "/teacher"

     if not (os.path.isdir(date_path)):
         os.mkdir(date_path)
     if not (os.path.isdir(mlp_path)):
         os.mkdir(mlp_path)
     if not (os.path.isdir(lstm_path)):
         os.mkdir(lstm_path)

     
     klts_training_loss_path = "{0}/klts_training_loss_2.npy".format(mlp_path)
     klst_training_loss_path  = "{0}/klst_training_loss.npy".format(mlp_path)
     klst_timestep_path = "{0}/klst_timestep.npy".format(mlp_path)
     klts_timestep_path = "{0}/klts_timestep.npy".format(mlp_path)
     klts_training_ret_path = "{0}/klts_training_ret_2.npy".format(mlp_path)
     klst_training_ret_path = "{0}/klst_training_ret.npy".format(mlp_path)

     lstm_dropout_ret_path = "{0}/lstm_ret_keep_prob_{1}.npy".format(lstm_path)
     lstm_dropout_loss_path = "{0}/lstm_dropout_prob_{1}.npy".format(lstm_path)

     lstm_ret_path = "{0}/lstm_ret.npy".format(lstm_path)
     lstm_loss_path = "{0}/lstm_losses.npy".format(lstm_path)
     teacher_ret_path = "{0}/teacher_ret.npy".format(teacher_path)


     plot( args.plot_all, args.plot_teacher, args.plot_klts, args.plot_klst, args.plot_lstm, args.plot_dropout )

