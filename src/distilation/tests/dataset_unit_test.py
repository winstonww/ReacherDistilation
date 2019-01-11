#!/usr/bin/env python
import sys
sys.path.append("/home/winstonww/reacher/src")
from distilation.config import *
from distilation.dataset import Dataset
import numpy as np
import json_tricks
import pdb

class DatasetTest(object):

    # ensure every
    def prev_pdflat_test(self):
        # test last episode
        episode = self.dataset.data_in_memory[-1]
        assert( np.array_equal(episode[0]["prev"], np.zeros([PDFLAT_SHAPE])) )
        for i in range(1, len(episode)):
            assert( np.array_equal(episode[i]["prev"], self.dataset.pdflat_at(episode,i-1)) )

        # test batch arrray
        self.dataset.curr_episode = self.dataset.data_in_memory[-1]
        prev_pdflat_batch_array = self.dataset.prev_pdflat_batch_array()
        print(prev_pdflat_batch_array)
        for i in range(0, STEPS_UNROLLED-1):
            assert(np.array_equal(prev_pdflat_batch_array[i,-1,:], self.dataset.curr_episode[-STEPS_UNROLLED+i+1]['prev']))
        assert(np.array_equal(prev_pdflat_batch_array[-1,-1,:], self.dataset.pdflat_at(self.dataset.curr_episode,self.dataset.last_step())) )

    # NOT USED NOT FEEDING PREVIOUS REWARD
    def prev_rew_test(self):
        # test last episode
        episode = self.dataset.data_in_memory[-1]
        assert( np.array_equal(episode[0]["prew"], np.zeros([1])) )
        for i in range(1, len(episode)):
            assert( np.array_equal(episode[i]["prew"], self.dataset.rew_at(episode,i-1)) )

        # test batch arrray
        self.dataset.curr_episode = self.dataset.data_in_memory[-1]
        prev_rew_batch_array = self.dataset.prev_rew_batch_array()
        print(prev_rew_batch_array)
        for i in range(0, STEPS_UNROLLED-1):
            assert(np.array_equal(prev_rew_batch_array[i,-1,:], self.dataset.curr_episode[-STEPS_UNROLLED+i+1]['prew']))
        assert(np.array_equal(prev_rew_batch_array[-1,-1,:], self.dataset.rew_at(self.dataset.curr_episode,self.dataset.last_step())) )



    def observation_test(self):
        # Test the function ob_batch_test_array()
        # Testing 3 scenarios: len(curr_episode) >/==/< STEPS_UNROLLED-1

        ob = np.array([-10 for i in range(OBSPACE_SHAPE) ])

        # Case 1:
        # len(curr_episode) < STEPS_UNROLLED-1
        # ob curr_episode = [ 2, 2 ]
        # expected ob_array = [0, 0, 2, 2, -10]
        if STEPS_UNROLLED > 5:
            self.dataset.curr_episode = self.dataset.data_in_memory[0][:STEPS_UNROLLED-3]
            ob_batch_test_array = self.dataset.ob_batch_test_array(ob)
            for i in range(0, STEPS_UNROLLED-(len(self.dataset.curr_episode)+1) ):
                assert( np.array_equal(ob_batch_test_array[i,-1,:], np.zeros([OBSPACE_SHAPE])) )
            j = 0
            for i in range(STEPS_UNROLLED-(len(self.dataset.curr_episode)+1), STEPS_UNROLLED-1):
                print(ob_batch_test_array[i,-1,:])
                print(self.dataset.curr_episode[j]["ob"])
                assert( np.array_equal(ob_batch_test_array[i,-1,:], self.dataset.curr_episode[j]["ob"]) )
                j += 1
            assert( np.array_equal(ob_batch_test_array[-1,-1,:], ob) )
        print(" Case 1 ok")

        # len(curr_episode) == STEPS_UNROLLED-1
        # Case 2
        # ob curr_episode = [ 2, 2, 2, 2 ]
        # expected ob_array = [ 2, 2, 2, 2, -10]
        self.dataset.curr_episode = self.dataset.data_in_memory[0][:STEPS_UNROLLED-1]

        ob_batch_test_array = self.dataset.ob_batch_test_array(ob)
        for i in range(0, STEPS_UNROLLED-1):
            assert( np.array_equal(ob_batch_test_array[i,-1,:], self.dataset.curr_episode[i]["ob"]) )
        assert( np.array_equal(ob_batch_test_array[-1,-1,:], ob) )
        print(" Case 2 ok")

        # Case 3 
        # len(curr_episode) > STEPS_UNROLLED-1
        # ob curr_episode = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ]
        # expected ob_array = [ 8, 9, 10, 11, -10]
        if STEPS_UNROLLED < 50:
            self.dataset.curr_episode = self.dataset.data_in_memory[0][:STEPS_UNROLLED+5]
            ob_batch_test_array = self.dataset.ob_batch_test_array(ob)
            j = len(self.dataset.curr_episode) - STEPS_UNROLLED + 1
            for i in range(STEPS_UNROLLED-1):
                assert( np.array_equal(ob_batch_test_array[i,-1,:], self.dataset.curr_episode[j]["ob"]) )
                j += 1
            assert( np.array_equal(ob_batch_test_array[-1,-1,:], ob) )
        print(" Case 3 ok")

    def holistic_test(self):
        ob = np.array([-10 for i in range(OBSPACE_SHAPE) ])
        self.curr_episode = self.dataset.data_in_memory[-1]
        print(self.dataset.ob_batch_test_array(ob))
        print(self.dataset.prev_pdflat_batch_array())
        print(self.dataset.data_in_memory[-1][-1])

    def length_test(self):
        print("data length is: ") 
        print(len(self.dataset.data_in_memory))
        for d in self.dataset.data_in_memory:
            print(len(d))
        pass


    def run(self):
        self.dataset = Dataset(dir_path="/home/winstonww/reacher/src/distilation/tests/")
        path = "/home/winstonww/reacher/data/20190110/033731/lstm/dataset_kp_1/dataset_1.json"
        #"/home/winstonww/reacher/src/distilation/tests/data/dataset.json"
        self.dataset.switch(page=path)
        self.observation_test()
        print("observation_test ok" )
        self.prev_pdflat_test()
        print("prev_pdflat_test ok" )
        self.holistic_test()
        print("holistic_test ok" )
        self.length_test()
        print("length test ok" )

test = DatasetTest()
test.run()
