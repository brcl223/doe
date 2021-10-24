import configparser as cfg
import os
from random import random, randint

import numpy as np


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
# Q-Learning Params
MAX_STEPS_PER_STATE = 10000
ALPHA = 0.01

TEST = False
if TEST:
    print("########################################")
    print("Warning: Test Mode On")
    print("########################################")
    MAX_STEPS_PER_STATE = 10
    TIMESTEPS = 3

def greedy_action(Q, s):
    actions = Q[s,:]
    return np.argmax(actions)


def eps_greedy_action(Q, s, eps):
    actions = Q[s,:]
    if eps > random():
        return randint(0, len(actions) - 1)
    
    return greedy_action(Q, s)


def step(T, R, s, a):
    state_val = random()
    state_prob = 0
    for sp in range(NUM_STATES):
        state_prob += T(s, a, sp)
        assert state_prob <= 1.02, f"State probability range beyond 1.\nValue: {state_prob}"
        if state_val <= state_prob:
            return sp, R(s, a, sp)

    assert False, "Step function returned no reward"


# To observe a terminal state, leave Qt as None
def observe(Q, s, a, sp, r, Q_next=None):
    next_state_reward = 0
    if Q_next is not None:
        max_action = greedy_action(Q_next, sp)
        next_state_reward = Q_next[sp, max_action]
    Q[s,a] += ALPHA * (r + GAMMA * next_state_reward - Q[s,a])


def main():
    _, T, R, _ = load_data(NAME)
    
    Qs = np.zeros((TIMESTEPS, NUM_STATES, NUM_ACTIONS))
    for i in reversed(range(TIMESTEPS)):
        Q_cur = Qs[i,:]
        Q_next = None
        if i + 1 < TIMESTEPS:
            Q_next = Qs[i+1,:]

        for s in range(NUM_STATES):
            for j in range(MAX_STEPS_PER_STATE):
                if j % 500 == 0:
                    print(f"[Timestep: {i}, State: {s}] Iter {j} of {MAX_STEPS_PER_STATE}...")
                eps = 1. - float(j) / MAX_STEPS_PER_STATE
                a = eps_greedy_action(Q_cur, s, eps)
                sp, r = step(T, R, s, a)

                observe(Q_cur, s, a, sp, r, Q_next=Q_next)

    # With Q^* calculated, calculate V^* to compare
    Vs = np.zeros((TIMESTEPS, NUM_STATES))
    policy = np.zeros((TIMESTEPS, NUM_STATES))
    for t in range(TIMESTEPS):
        Q_cur = Qs[t,:]
        for s in range(NUM_STATES):
            max_action = greedy_action(Q_cur, s)
            policy[t,s] = max_action
            Vs[t,s] = Q_cur[s,max_action]

    # Format Q to be saved as 1D array
    Qs = Qs.flatten()

    np.savetxt(f"./policies/q-learning/{NAME}.q-star", Qs)
    np.savetxt(f"./policies/q-learning/{NAME}.v-star", Vs.T)
    np.savetxt(f"./policies/q-learning/{NAME}.policy", policy.T)


NAME = "advanced"
CONFIG = Config(NAME)
CFG = CONFIG.fields
NUM_STATES = int(CFG["MDP"]["states"])
NUM_ACTIONS = int(CFG["MDP"]["actions"])
TIMESTEPS = int(CFG["MDP"]["timesteps"])
GAMMA = float(CFG["MDP"]["gamma"])


if __name__ == '__main__':
    main()
