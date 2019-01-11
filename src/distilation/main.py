#!/usr/bin/env python
import sys
sys.path.append("/home/winstonww/reacher/src")
import argparse
from distilation import lstm_train, mlp_train
from distilation.config import DATE, KEEP_PROB

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("-lt"     ,"--lstm_train", help="train lstm", action="store_true")
     parser.add_argument("-ct"     ,"--mlp_train", help="train mlp", action="store_true")
     parser.add_argument("-k"     ,"--keep_prob", help=" keep_prob on lstm ob dropout ", nargs=1, default=None )
     parser.add_argument("-ch"     ,"--check", help="check point", action="store_true")
     parser.add_argument("-r"     ,"--restore", help="restore", action="store_true")
     args = parser.parse_args()

     if args.keep_prob:
         global KEEP_PROB
         KEEP_PROB = args.keep_prob

     if args.check:
         print( " checking saved variables " )
         chkp.print_tensors_in_checkpoint_file( lstm_trained_data_path, tensor_name='', all_tensors=True, all_tensor_names=True)
     elif args.lstm_train:
         lstm_train.train( True, args.restore )
     elif args.mlp_train:
         mlp_train.train( True, args.restore )
