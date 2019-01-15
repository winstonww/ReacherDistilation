#!/usr/bin/env python
import os
import datetime
from pathlib import Path
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
import re

def load_file(path):
    arr = []
    if not os.path.exists(path):
        print("Source npy does not exists @ {0}".format(path))
        # warnings.warn("Source npy does not exists")
    else:
        arr = np.load(path)
    return arr

def get_data( filename, plot_loss ):
    factor = 1 if plot_loss else 50
    data = load_file( filename )[ STARTING_EPISODE:ENDING_EPISODE ]
    data = [ sum( data[ i:i+SLIDING_WINDOW ] ) / ( SLIDING_WINDOW * factor )
        for i in range( 0, ENDING_EPISODE-STARTING_EPISODE, SLIDING_WINDOW ) ]
    return data

def plot( date_path ):
    
    metadata = get_plot_metadata( date_path )

    for metadatum in metadata: 

        data = get_data( metadatum[ 'input' ], metadatum[ 'ylabel' ] == 'KL Divergence' or metadatum[ 'ylabel' ] == 'l2 Loss of value function prediction' )

        plt.figure( figsize=( 10, 8 ) )
        plt.xlabel( metadatum[ 'xlabel' ] )
        plt.ylabel( metadatum[ 'ylabel' ] )
        plt.title( metadatum[ 'title' ] )
        plt.plot(data)
        plt.savefig( metadatum[ 'output' ], dpi=250 )
        print( metadatum[ 'output' ] )

def plot_by_data( date_path ):
    metadata = get_plot_metadata( date_path )
    
    losses, rets = [], []

    for metadatum in metadata:
        if metadatum[ 'ylabel' ] ==  'Average reward':
            rets.append( metadatum )
        elif metadatum[ 'ylabel' ] ==  'KL Divergence':
            losses.append( metadatum )

    plot_type( losses, date_path )
    plot_type( rets, date_path )


def plot_type( metadata, date_path ):
    plt.figure( figsize=( 10, 8 ) )
    plt.xlabel( metadata[0][ 'xlabel' ] )
    plt.ylabel( metadata[0][ 'ylabel' ] )
    title = "{0} averaging every {1} epsiodes for a total of {2} episodes".format( 
            metadata[0][ 'ylabel' ], SLIDING_WINDOW, ENDING_EPISODE - STARTING_EPISODE )
    plt.title( title )
    for metadatum in metadata:
        data = get_data( metadatum[ 'input' ], metadatum[ 'ylabel' ] == 'KL Divergence' )
        plt.plot( data, label=metadatum[ 'agent' ] )
    plt.legend()

    print( "File Name" )
    print("_".join( metadata[0][ 'ylabel' ].lower().split(' ') ) + ".png" ) 
    plt.savefig( date_path + "/" + "_".join( metadata[0][ 'ylabel' ].lower().split(' ') ) + ".png" , dpi=250 )





def get_plot_metadata( date_path ):

    agent_abrev = { 'KL': 'KL Divergence', 'ret': 'Average reward' }

    #  get all npy files
    filenames = [ filename for filename in glob.iglob("{0}/**/*.npy".format( date_path ), recursive=True) ]

    metadata = []
    for filename in filenames:
        match = re.search( r'.*/(.*)_(KL|ret).npy', filename, re.M|re.I )
        if match:
            agent = " ".join( match.group(1).split("_") )
            xlabel = "Every {0} episode(s)".format( SLIDING_WINDOW )
            ylabel = agent_abrev[ " ".join( match.group(2).split("_") ) ]
            title = "{0} of {1} averaging every {2} epsiodes for a total of {3} episodes".format( 
                    ylabel, agent, SLIDING_WINDOW, ENDING_EPISODE - STARTING_EPISODE )

            output = "{0}/{1}/{2}_{3}_sliding_window_{4}.png".format( 
                date_path, "lstm" if "lstm" in agent else "mlp", match.group(1), match.group(2), SLIDING_WINDOW )

            metadata.append( { 'title':  title, 
                               'xlabel': xlabel, 
                               'ylabel': ylabel, 
                               'agent':  agent, 
                               'input':  filename,
                               'output': output } )
    return metadata
        
           


if __name__ == '__main__':

    date = datetime.datetime.now().strftime( "%Y%m%d" )
    parser = argparse.ArgumentParser()
    #parser.add_argument("-lt"     ,"--lstm_train", help="train lstm", action="store_true")
    parser.add_argument("-s"     ,"--starting", help="starting episode", nargs=1, default=[ 0 ] )
    parser.add_argument("-e"     ,"--ending", help=" keep_prob on lstm ob dropout ", nargs=1, default=[ 5000 ] )
    parser.add_argument("-w"     ,"--sliding_window", help="starting episode", nargs=1, default=[ 1 ] )
    parser.add_argument("-d"     ,"--date", help="custom date", nargs=1, default=[ date ] )
    args = parser.parse_args()

    STARTING_EPISODE = int( args.starting[0] )
    ENDING_EPISODE = int( args.ending[0] )
    SLIDING_WINDOW = int( args.sliding_window[0] )

    base_path =  "{0}/reacher/data".format( str(Path.home()) )
    date_path = "{0}/{1}".format(base_path, args.date[0] )
    plot( date_path )
    plot_by_data( date_path )

    
