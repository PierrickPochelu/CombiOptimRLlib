from typing import *
import copy
import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Space, Discrete, Box
from JobScheduling import util
from JobScheduling.util import Item

VERBOSE = False

def log(*args):
    if VERBOSE:
        print(*args)

MAX_WEIGHT=4
MAX_DURATION=4
CAPACITY = 5
NB_ITEMS = 9
SUBMISSION_DEADLINE=27 # Items are submitted randomly from 0 to `SUBMISSION`


BONUS_JOB_IS_DONE = 1000
REWARD_PENALTY=1000 # hard constraint violated

# A-B**2
REWARD_PENALTY_PENDING_ITEM_A=0.1
REWARD_PENALTY_PENDING_ITEM_B=0.1



# ITEM GENERATORS
ITEMS=[]
for i in range(NB_ITEMS):
    submitted_date=np.random.randint(1, SUBMISSION_DEADLINE)
    weight=np.random.randint(1,MAX_WEIGHT)
    duration=np.random.randint(1,MAX_DURATION)
    I=Item(start=-1, duration=duration, weight=weight, identifier=i, submitted_date=submitted_date)
    ITEMS.append( I )
ITEMS=sorted(ITEMS, key=lambda x:x.submitted_date, reverse=True) # Pop(-1) is O(1) so we sort them in reverse order

COMPAR_FOR_BACKFILLING_SELECT=[util.cond_biggest, util.cond_smallest,
                                  util.cond_longest, util.cond_shortest,
                                  util.cond_latest]

def inv_prepro_state(prep_state):
    return prep_state

def _prepro_weight(w):
    return w/MAX_WEIGHT # 0 value is reserved when no object is incoming
def _prepro_duration(d):
    return d/MAX_DURATION
def _prepro_resource(r:np.array):
    return r/CAPACITY

def prepro_state(raw_state):
    return raw_state

def prepro_reward(raw_reward):
    prep_reward = raw_reward
    return prep_reward

def inv_prepro_reward(prep_action):
    raw_action = prep_action
    return raw_action

class EnvKnapSack(gym.Env):
    def __init__(self, env_config, seed=42):
        np.random.seed(seed)

        self.action_type = env_config["action_type"]  # "continuous" or "discrete"


        # observation space
        self.Ns = 17 #current job + backfilling embedding + 1 backfilling size + remaining ressources embedding
        self.observation_space = Box(low=0., high=1., shape=(self.Ns,), dtype=np.float32)

        # Action space
        self.Na = 2 + len(COMPAR_FOR_BACKFILLING_SELECT) #nothing, current job , backfilling embedding
        if self.action_type == "discrete":
            self.action_space = Discrete(self.Na)
        elif self.action_type == "continuous":
            raise ValueError("Not excepted action_type")
            #self.action_space = Box(0., 1., (self.Na,))
        else:
            raise ValueError("action type not understood. Should be in {discrete,continous}")
        self.metadata = {"render.modes": ["human"]}


        self.init_reward = 0
        self.init_done = False
        self.init_info = {"incoming_item":None,
                           "pending_items":[],
                           "available_resource": np.zeros((MAX_DURATION,), dtype=float)}
        self.init_round = 0
        self.init_items = ITEMS.copy() # items are already sorted according submitted_date


        self.ss=util.SchedulingSim(CAPACITY)

        self.round = None # round is the item id
        self.items=None
        self.info=None


        # Set with initial state
        # "state" mentions the raw state
        self.info, self.reward, self.done, self.info= None, None, None, None


        self.reset(testing=False)

        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            log("ERROR : INVALID INIT STATE (bounds or type)", self.state)

    def compute_state_for_rl(self, info):
        pending_items = info["pending_items"]
        avail_resource = info["available_resource"]
        incoming_item = info["incoming_item"]

        # incoming item
        if incoming_item is None:
            embedding_incoming_item = np.zeros((2,), dtype=np.float32)
        else:
            embedding_incoming_item = np.array(
                [_prepro_weight(incoming_item.weight), _prepro_duration(incoming_item.duration)], dtype=np.float32)
        assert (len(embedding_incoming_item) == 2)

        # available resource in t+0, t+1, t+2 ... t+`CAPACITY`
        avail_resource_embedding = _prepro_resource(avail_resource)
        assert (len(avail_resource_embedding) == 4)

        # Give some pending items info: the latest, the biggest , the smallest, the shortest, the longest ...
        # It allows to reduce observation_space
        embedding_pending_item = np.zeros((len(COMPAR_FOR_BACKFILLING_SELECT) * 2,),
                                          dtype=np.float32)
        sorted_pending_items = sorted(pending_items, key=lambda x: x.submitted_date)

        i = 0
        for f in COMPAR_FOR_BACKFILLING_SELECT:
            PI = util.get_fitting_item_according_cond(sorted_pending_items, CAPACITY-avail_resource[0], f)
            if PI is not None:
                embedding_pending_item[i] = _prepro_weight(PI.weight)
                embedding_pending_item[i + 1] = _prepro_duration(PI.duration)
                i += 2
        assert(len(embedding_pending_item)==10)

        # global stat on pending items
        embedding_pending_size = np.array([float(len(sorted_pending_items)) / NB_ITEMS], dtype=np.float32)
        # %TODO give more info: mean(weight), std(weight), mean(duration), std(duration)
        assert (len(embedding_pending_size) == 1)

        # concatenate embeddings
        prep_state = np.concatenate([embedding_incoming_item,
                                     avail_resource_embedding,
                                     embedding_pending_item,
                                     embedding_pending_size]).astype(np.float32)
        return prep_state

    def _transition(self, action):
        """
        Update state,info,reward,done
        :param action:
        :return:
        """
        pending_job_penalty = util.starvation_penalty(self.info["pending_items"], self.round,
                                                      REWARD_PENALTY_PENDING_ITEM_A, REWARD_PENALTY_PENDING_ITEM_B,
                                                      REWARD_PENALTY)

        # From (incoming_item, action) -> (action_penalty, done)
        I = self.info["incoming_item"]
        if action == 0:  # Nothing
            # if there is an item -> backfilling
            # if there is no item -> next timestep
            if I is None:
                self.round += 1 # next step
            else:
                self.info["pending_items"].append(I)
            action_penalty=0
            done=False
        elif action == 1:  # Scheduling the incoming item
            if I is not None and self.ss.can_fit(self.round, I.weight):
                # Schedule it
                I.start=self.round
                self.ss.add_item(I)
                action_penalty=0
                done=False
            else:
                # Crash we do no have enough resource OR I is None
                action_penalty = -REWARD_PENALTY
                done = True
        else:
            a=action - 2  # the biggest fitting (any duration). When equality, takes the latest.
            comp=COMPAR_FOR_BACKFILLING_SELECT[a]
            BFI=util.get_fitting_item_according_cond(self.info["pending_items"],
                                                   self.info["available_resource"][0],
                                                   comp)

            if BFI is None:
                # Unvalid action
                action_penalty = -REWARD_PENALTY
                done = True
            else:
                # Launch BFI
                done=False
                action_penalty=0
                self.info["pending_items"].remove(BFI)  # Warning: complexity O(n)
                BFI.start=self.round
                self.ss.add_item(BFI) #go

            # Store I in pending items
            if I is not None:
                self.info["pending_items"].append(I)

        # Round consraint for accelerating purpose
        round_penalty=0
        if self.round >= SUBMISSION_DEADLINE+MAX_DURATION*NB_ITEMS:
            done=True
            round_penalty=REWARD_PENALTY

        # Done
        bonus_job_is_done=0
        if done==True:
            self.done=True
        else:
            self.done = len(self.items)==0 and len(self.info["pending_items"])==0 and self.info["available_resource"][0]==0
            if self.done:
                bonus_job_is_done=BONUS_JOB_IS_DONE



        # Reward
        self.reward = action_penalty + round_penalty + pending_job_penalty + bonus_job_is_done

        # TODO: code to edit ?
        # re-sorted items (after action)
        self.items=sorted(self.items, key=lambda x:x.submitted_date, reverse=True)
        if len(self.items)>0 and self.items[-1].submitted_date==self.round:
            self.info["incoming_item"] = self.items.pop() # next item
        else:
            self.info["incoming_item"] = None

        # Compute future ressource
        fixed_size=np.zeros((MAX_DURATION,))
        future=self.ss.timeline_ressource[self.round: self.round + CAPACITY] # can be size 1, 2,...
        fixed_size[:len(future)]=future
        self.info["available_resource"]=fixed_size

        # Additional information for debugging purpose
        if VERBOSE:
            self.info["round"]=self.round

        self.state=self.compute_state_for_rl(self.info)
        return self.state, self.info, self.reward, self.done

    def step(self, action):
        try:
            assert (self.action_space.contains(action))
        except AssertionError:
            log("WARNING : INVALID ACTION", action)

        log("***************")
        log("Info before:")
        log(self.info)
        log("Action:", action)
        self.state, self.info, self.reward, self.done = self._transition(action)
        log("Info after: ")
        log(self.info)
        log(f"reward:{self.reward} done:{self.done}")
        log("***************")
        # round management:
        # do we need to iterate on current objects ?
        # if action==0 and no object get self.ss.get_items(t)

        # Check state
        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            log("WARNING : INVALID STATE", self.state)

        # Prepo for ML algo.
        prep_state = prepro_state(self.state)
        prep_reward = prepro_reward(self.reward)
        return prep_state, prep_reward, self.done, self.info

    def reset(self, testing=False):
        self.ss=util.SchedulingSim(CAPACITY)
        self.round = self.init_round
        self.items = copy.deepcopy(self.init_items)


        self.reward = self.init_reward
        self.done = self.init_done
        self.info = copy.deepcopy(self.init_info)

        # first incoming item
        if self.items[-1].submitted_date==self.round:
            first_item=self.items.pop()
            self.info["incoming_item"]=first_item
        else:
            self.info["incoming_item"] = None


        self.state = self.compute_state_for_rl(self.info) #use info
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
