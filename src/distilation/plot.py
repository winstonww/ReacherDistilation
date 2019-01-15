#!/usr/bin/env python
import re
import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.append("/home/winstonww/reacher/src")
import datetime
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt
from distilation.utils import load_file


filepaths = [
"/home/winstonww/reacher/data/kp1.0.npy",
"/home/winstonww/reacher/data/kp0.85.npy",
"/home/winstonww/reacher/data/kp0.75.npy",
"/home/winstonww/reacher/data/kp0.75s.npy",
"/home/winstonww/reacher/data/kp0.5.npy",
#"/home/winstonww/reacher/data/kp0.5ss.npy",
"/home/winstonww/reacher/data/kp0.2.npy",
"/home/winstonww/reacher/data/kp0.1.npy",
"/home/winstonww/reacher/data/kp0.05.npy",
"/home/winstonww/reacher/data/kp0.0.npy",
]

def plot():
    # Sets up the plot
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Episode no. (x10)")
    ax.set_ylabel("Average reward")
    ax.set_title("Average student reward of every 50 episodes (3000 total)")
    ax.axis
    for filepath in filepaths:
        kp = re.match(".*kp(\d+\.\d+\w*).npy",filepath).group(1)
        data = load_file( filepath )
        ax.plot(data[0:60], label="kp={0}".format(kp))
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([-0.30,-0.08])
    plt.savefig("/home/winstonww/reacher/data/ret.png", dpi=250)

plot()
