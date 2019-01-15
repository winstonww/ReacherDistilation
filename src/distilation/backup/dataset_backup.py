import numpy as np
import random
from distilation.config import *
import json_tricks
from os import listdir
from os.path import isfile, join

__all__ = ['Dataset', 'DatasetStore']

'''
To handle large datasets, this store class manages memory and
splits the dataset into multiple pages for efficient storage and retrieval.
'''
class DatasetStore(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.pages = self.collect_pages(dir_path)
        self.curr_page = self.get_full_path(0)

    @property
    def curr_page(self):
        return self.__curr_page

    @curr_page.setter
    def curr_page(self, c):
        self.__curr_page = c

    def get_full_path(self,page):
        return self.dir_path+"/dataset_"+str(len(self.pages))+".json"

    def store(self, data_in_memory):
        with open(self.curr_page, 'wb+') as fh:
            json = json_tricks.dumps(obj=data_in_memory, compression=True, primitives=True)
            fh.write(json)

        full = len(data_in_memory) >= MAX_CAPACITY
        if full: 
            self.curr_page = self.create_new_page()
            return list()
        return data_in_memory

    # this load page does not modifty self.data_in_memory, 
    # just return the data_in_memory stored at the given page
    def load(self,page):
        with open(page, "rb+") as fh:
            json = fh.read()
        data_in_memory = json_tricks.loads(json, decompression=True)
        return data_in_memory

    # returns a random page from pages
    def rand_pages(self, num_pages):
        if not self.pages: return None
        return random.sample( self.pages, min(num_pages,len(self.pages)) )

    def create_new_page(self):
        self.pages.append( self.curr_page )
        page = self.get_full_path( len(self.pages) )
        if os.path.exists(page):
            raise FileExistsError("current page already exists. will not overwrite")
        print("new page path @ {0}".format(page) )
        return page

    # this function gets all pages in the given directory
    def collect_pages(self,dir_path):
        return [join(dir_path,f) for f in listdir(dir_path) if isfile(join(dir_path, f))]



'''
This is custom dataset class used for training lstm student nn.
'''
class Dataset(object):
    def __init__(self, dir_path):
        self.dstore = DatasetStore(dir_path=dir_path)
        self.data_in_memory = []
        self.curr_episode = []
        self.num_total_episodes = 0
        self.training_data = []

    def dump(self):
        try:
            self.data_in_memory = self.dstore.store(self.data_in_memory)
        except FileExistsError as error:
            print("Can't create new page")
            print(error)
    
    def pages(self):
        import re
        idx =[]
        for page in self.dstore.pages:
            m = re.search("dataset_(\d+).json", page)
            idx.append( int(m.group(1)) )
        z = zip( self.dstore.pages, idx )
        a = sorted(z, key=lambda e: e[1] )
        sorted_pages, _ = zip(*a)
        return sorted_pages

    @property
    def data_in_memory(self):
        return self.__data_in_memory

    @data_in_memory.setter
    def data_in_memory(self, d):
        self.__data_in_memory = d

    @property
    def curr_episode(self):
        return self.__curr_episode

    @curr_episode.setter
    def curr_episode(self, d):
        self.__curr_episode = d

    def switch(self,page):
        self.data_in_memory = self.dstore.load(page)
        #print("Page switched to {0}".format(page))

    def write(self,
            ob=np.zeros([OBSPACE_SHAPE]),
            reward=0,
            t_pdflat=np.zeros([PDFLAT_SHAPE]),
            s_pdflat=np.zeros([PDFLAT_SHAPE]),
            stepped_with='t'):

            # accumulate teacher action data
            step = {}
            step["ob"] = np.squeeze(ob)
            step["rew"] = np.expand_dims(reward,axis=0)
            step['t'] = np.squeeze(t_pdflat)
            step['s'] = np.squeeze(s_pdflat)
            step["with"] = stepped_with
            step["prev"] = self.pdflat_at(self.curr_episode, self.last_step())
            step["prew"] = self.rew_at(self.curr_episode, self.last_step())

            #print("")
            #print("teacher curr action:")
            #print( t_pdflat)
            #print("student curr action:")
            #print( s_pdflat)
            #print("prev action:")
            #print( step["prev"] )

            self.curr_episode.append(step)

    
    def flush(self):
        self.data_in_memory.append(self.curr_episode)
        self.curr_episode = []
        self.num_total_episodes += 1

    # For teacher foring
    def pdflat_at(self, episode, timestep, use_student=False):
        # set action to zero
        if timestep < 0: return np.zeros([PDFLAT_SHAPE])
        if not use_student:
            return episode[timestep]['t'] 
        #if episode[timestep]['with'] == 't':
        #    return episode[timestep]['t'] 
        #return episode[timestep]['s']

    def rew_at(self, episode, timestep):
        # set action to zero
        if timestep < 0: return [0]
        return episode[timestep]['rew']


    def reset_training_data(self ):
        self.training_data = self.data_in_memory[:]
        if not self.dstore.pages:
            #print("training data set to current page")
            return 

        rand_pages = self.dstore.rand_pages(10)
        for rand_page in rand_pages:
            if rand_page and rand_page != self.dstore.curr_page:
                #print("training data includes {0}".format(rand_page))
                self.training_data.extend(self.dstore.load(page=rand_page))

    # training batch
    def training_batches(self):

        if ( len(self.curr_episode) % 25 == 0 ) or not self.training_data:
            self.reset_training_data()

        for i in range(TRAINING_EPOCHS):

            episodes = [ random.choice(self.training_data) for _ in range(LSTM_BATCH_SIZE) ] 
            start = random.randint( 0, EPISODE_STEPS - STEPS_UNROLLED  ) 

            ob_batch_array            = self.serialize("ob", episodes, start, start+STEPS_UNROLLED)
            t_pdflat_batch_array      = self.serialize('t', episodes, start, start+STEPS_UNROLLED)
            prev_pdflat_batch_array   = self.serialize("prev", episodes, start, start+STEPS_UNROLLED)
            prev_rew_batch_array   = self.serialize("prew", episodes, start, start+STEPS_UNROLLED)

            yield (ob_batch_array, t_pdflat_batch_array, prev_pdflat_batch_array, prev_rew_batch_array)

    # returns a np array of data of datatype; 
    def serialize(self, datatype, episodes, start, end):
        outer = list()
        for episode in episodes:
            inner = np.array(self._serialize(datatype, episode, start, end))
            outer.append(inner)
        return np.transpose(np.array(outer), (1,0,2))

    # a helper to serialize(); return a list of data of type datatype in the given episode 
    # end is non-inclusive
    def _serialize(self, datatype, episode, start, end):
        inner = list()
        for i in range(start,end):
            inner.append(episode[i][datatype])
        return inner

    # test batch; return ob_batch_array given the current observation and current episode
    def test_batch(self,ob):
        ob_batch_array = self.ob_batch_test_array(ob)
        prev_pdflat_batch_array = self.prev_pdflat_batch_array()
        prev_rew_batch_array = self.prev_rew_batch_array()
        return ob_batch_array, prev_pdflat_batch_array, prev_rew_batch_array

    def ob_batch_test_array(self, ob):

        curr_episode_obs = list()
        ob_batch_array = np.zeros([ STEPS_UNROLLED, LSTM_BATCH_SIZE, OBSPACE_SHAPE ])

        # example:
        # len(curr_episode) = 5
        # STEPS_UNROLLED = 3
        # So, start_idx = 3
        start = (len(self.curr_episode) - STEPS_UNROLLED + 1)

        # if start index < 0, pad observations with equal number of zeros first at the beginning
        if start < 0: 
            curr_episode_obs.extend( [ np.zeros([OBSPACE_SHAPE]) for i in range(0-start) ] )
            start = 0

        curr_episode_obs.extend(self._serialize("ob", self.curr_episode, 
                start, len(self.curr_episode)))

        curr_episode_obs.append( ob )

        #set last batch to last_batch
        ob_batch_array[:,LSTM_BATCH_SIZE-1,:] = np.array(curr_episode_obs)
        return ob_batch_array

    def prev_pdflat_batch_array(self):
        prev_pdflat_array = np.zeros([ STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE ] )
        prev_pdflats = list()
        start = (len(self.curr_episode) - STEPS_UNROLLED + 1)

        # if start index < 0, pad observations with equal number of zeros first at the beginning
        if start < 0: 
            prev_pdflats.extend( [ np.zeros([PDFLAT_SHAPE]) for i in range(0-start) ] )
            start = 0

        prev_pdflats.extend(self._serialize("prev",
                self.curr_episode, start, len(self.curr_episode)) )

        prev_pdflats.append( self.pdflat_at(self.curr_episode, self.last_step()) )

        prev_pdflat_array[:,LSTM_BATCH_SIZE-1,:] = np.array(prev_pdflats)
        return prev_pdflat_array

    def prev_rew_batch_array(self):
        prev_rew_array = np.zeros([ STEPS_UNROLLED, LSTM_BATCH_SIZE, 1 ] )
        prev_rews = list()
        start = (len(self.curr_episode) - STEPS_UNROLLED + 1)

        # if start index < 0, pad observations with equal number of zeros first at the beginning
        if start < 0: 
            prev_rews.extend( [ [0] for i in range(0-start) ] )
            start = 0

        prev_rews.extend(self._serialize("prew",
                self.curr_episode, start, len(self.curr_episode)) )

        prev_rews.append( self.rew_at( self.curr_episode, self.last_step() ) )

        prev_rew_array[:,LSTM_BATCH_SIZE-1,:] = np.array(prev_rews)
        return prev_rew_array

    def num_episodes(self):
        return self.num_total_episodes
    
    def last_step(self):
        return len(self.curr_episode)-1
