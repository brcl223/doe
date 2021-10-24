import configparser as cfg
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# .common.py because pythom imports are a pain
class DataTable:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.data = np.zeros((states,actions,states))

    def __call__(self, s, a, sp):
        return self.data[sp,a,s]


class DataVec:
    def __init__(self, states):
        self.states = states
        self.data = np.zeros(states)

    def __call__(self, s):
        return self.data[s]

    def __setitem__(self, i, val):
        self.data[i] = val

    def __len__(self):
        return len(self.data)

    def init_clone(self):
        return DataVec(self.states)


class Callable:
    def __init__(self, v):
        self.v = v

    def __call__(self, i):
        return self.v[i]


class Config:
    def __init__(self, name):
        self.base_path = f"./transitions/{name}/"
        self.fields = load_config(self.base_path)


### Data loading utilities
def load_policies(ty="closed"):
    policies = []
    for policy in glob.glob(f"./policies/{ty}-loop/*.policy"):
        name = re.search(r"\./policies/(.*)\.policy", policy)
        p = (name[1], np.loadtxt(policy))
        policies.append(p)
    return policies


def load_data(name):
    cfg = Config(name)
    T, R, ns = load_table_data(cfg)
    V = DataVec(ns)
    return cfg, T, R, V


def load_config(base_path):
    config = cfg.ConfigParser()
    path = os.path.join(base_path, "metadata.ini")
    config.read(path)
    return config


def load_table_data(cfg):
    NUM_STATES = int(cfg.fields["MDP"]["states"])
    NUM_ACTIONS = int(cfg.fields["MDP"]["actions"])
    T = DataTable(NUM_STATES, NUM_ACTIONS)
    R = DataTable(NUM_STATES, NUM_ACTIONS)
    for i in range(NUM_ACTIONS):
        tpath = os.path.join(cfg.base_path, f"a{i+1}.csv")
        rpath = os.path.join(cfg.base_path, f"r{i+1}.csv")
        T.data[:,i,:] = np.genfromtxt(tpath, delimiter=',')
        R.data[:,i,:] = np.genfromtxt(rpath, delimiter=',')
    return T, R, NUM_STATES



######################################
# Start of actual script
######################################
def main():
    cfg, T, R, V = load_data(NAME)
    Vs = np.zeros((NUM_STATES, TIMESTEPS))
    for i in range(TIMESTEPS):
        Vcur = V.init_clone()
        # s = Current state
        for s in range(NUM_STATES):
            val,_ = calculate_value(T,R,V,s)
            Vcur[s] = val
        V = Vcur
        Vs[:,i] = V.data[:]

    policy = np.zeros((NUM_STATES, TIMESTEPS))
    for i in range(policy.shape[1]):
        Vcur = Callable(Vs[:,i])
        for s in range(NUM_STATES):
            _,action = calculate_value(T,R,Vcur,s)
            policy[s,i] = action

    # Flip to make time easier to read
    policy = np.flip(policy,1)
    Vs = np.flip(Vs,1)

    np.savetxt(f"./policies/value-iter/{NAME}.policy", policy)
    np.savetxt(f"./policies/value-iter/{NAME}.v-star", Vs)

    # Plot data
    # fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
    # im_policy = ax1.imshow(policy)
    # im_value = ax2.imshow(Vs)
    # ax2.figure.colorbar(im_value, ax=ax2)

    # ax1.set_ylabel("State")
    # ax1.set_xlabel("Timestep")
    # ax2.set_ylabel("State")
    # ax2.set_xlabel("Timestep")

    # timesteps = [str(x) for x in range(2,11,2)]
    # ax1.set_xticks(np.arange(len(timesteps)))
    # ax2.set_xticks(np.arange(len(timesteps)))
    # ax1.set_yticks(np.arange(NUM_ACTIONS))
    # ax2.set_yticks(np.arange(NUM_ACTIONS))

    # states = ["Wear 1", "Wear 2", "Wear 3", "Broken"]
    # actions = ["CS1", "CS2", "CS3", "Rep"]
    # ax1.set_yticklabels(states)
    # ax2.set_yticklabels(states)
    # ax1.set_xticklabels(timesteps)
    # ax2.set_xticklabels(timesteps)

    # ax1.set_title("Policy per Timestep")
    # ax2.set_title("State Value per Timestep")

    # for i in range(NUM_STATES):
    #     for j in range(TIMESTEPS//CHECKPOINT):
    #         ax1.text(j, i, actions[int(policy[i,j])], ha="center", va="center", color="w")

    # # fig.suptitle("Wear Based Reward / High Negative Penalty")
    # plt.show()
    
    # fig, ax = plt.subplots()
    # im_policy = ax.imshow(policy[:,1::2])

    # ax.set_ylabel("State")
    # ax.set_xlabel("Timestep")

    # timesteps = [str(x) for x in range(2,11,2)]
    # ax.set_xticks(np.arange(len(timesteps)))
    # ax.set_yticks(np.arange(NUM_ACTIONS))

    # states = ["Wear 1", "Wear 2", "Wear 3", "Broken"]
    # actions = ["CS1", "CS2", "CS3", "Rep"]
    # ax.set_yticklabels(states)
    # ax.set_xticklabels(timesteps)

    # ax.set_title("Aerospace Manufacturing Policy")

    # for i in range(NUM_STATES):
    #     for j in range(TIMESTEPS//CHECKPOINT):
    #         ax.text(j, i, actions[int(policy[i,j])], ha="center", va="center", color="w")

    # fig.suptitle("Automotive Manufacturing")
    # ax.title = "Automotive Manufacturing"
    # plt.show()

def calculate_value(T, R, V, s):
    max_action = 0
    max_val = float("-inf")
    # a = actions
    for a in range(NUM_ACTIONS):
        cur_val = 0
        # sp = s' (next state)
        for sp in range(NUM_STATES):
            cur_val += T(s,a,sp)*(R(s,a,sp)+GAMMA*V(sp))
        if cur_val > max_val:
            max_val = cur_val
            max_action = a
    return max_val, max_action


def calculate_policy(T, R, V):
    policy = np.zeros(len(V))
    for s in range(NUM_STATES):
        _, action = calculate_value(T,R,V,s)
        policy[s] = action
    return policy


NAME = "basic"
CONFIG = Config(NAME)
CFG = CONFIG.fields
NUM_STATES = int(CFG["MDP"]["states"])
NUM_ACTIONS = int(CFG["MDP"]["actions"])
TIMESTEPS = int(CFG["MDP"]["timesteps"])
CHECKPOINT = int(CFG["MDP"]["timesteps_per_eval"])
GAMMA = float(CFG["MDP"]["gamma"])

if __name__ == '__main__':
    main()
