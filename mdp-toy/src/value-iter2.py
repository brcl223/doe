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
    cfg, T, R, _ = load_data(NAME)
    Vs = np.zeros((TIMESTEPS, NUM_STATES))
    policy = np.zeros((TIMESTEPS, NUM_STATES))
    for t in reversed(range(TIMESTEPS)):
        policy_cur = policy[t,:]
        V_cur = Vs[t,:]
        V_next = None
        if t + 1 < TIMESTEPS:
            V_next = Vs[t+1,:]

        for s in range(NUM_STATES):
            max_val, max_action = calculate_value(T, R, V_cur, s, V_next=V_next)
            V_cur[s] = max_val
            policy_cur[s] = max_action

    np.savetxt(f"./policies/value-iter/{NAME}.policy", policy.T)
    np.savetxt(f"./policies/value-iter/{NAME}.v-star", Vs.T)


def calculate_value(T, R, V, s, V_next=None):
    max_action = 0
    max_val = float("-inf")
    # a = actions
    for a in range(NUM_ACTIONS):
        cur_val = 0
        # sp = s' (next state)
        for sp in range(NUM_STATES):
            next_state_reward = 0
            if V_next is not None:
                next_state_reward = GAMMA * V_next[sp]
            cur_val += T(s,a,sp) * (R(s,a,sp) + next_state_reward)
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


NAME = "advanced"
CONFIG = Config(NAME)
CFG = CONFIG.fields
NUM_STATES = int(CFG["MDP"]["states"])
NUM_ACTIONS = int(CFG["MDP"]["actions"])
TIMESTEPS = int(CFG["MDP"]["timesteps"])
CHECKPOINT = int(CFG["MDP"]["timesteps_per_eval"])
GAMMA = float(CFG["MDP"]["gamma"])

if __name__ == '__main__':
    main()
