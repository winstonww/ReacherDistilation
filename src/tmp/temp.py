#!/usr/bin/env python
from pathlib import Path
import os
import numpy as np
def load_file(path):
    arr = []
    if not os.path.exists(path):
        print("Source npy does not exists")
        # warnings.warn("Source npy does not exists")
    else:
        arr = np.load(path)
    return arr

if __name__ == '__main__':
     base_path =  "{0}/reacher/data".format( str(Path.home()) )
     date_path = "{0}/{1}".format(base_path, '20180901' )
     mlp_path = date_path + "/mlp"
     lstm_path = date_path + "/lstm"
     ob_list_path =                           "{0}/lstm_with_keep_probability_{1}_ob.npy".format( lstm_path, str( 0.7 ) )
     print( len( load_file( ob_list_path ) ) )

