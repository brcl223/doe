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
ALPHA = 0.01
TIMESTEPS = 10
GAMMA = 0.99

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
    num_states = T.data.shape[0]
    for sp in range(num_states):
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


def qlearn_policy(model, max_steps_per_state, Qs=None):
    _, T, R, _ = load_data(model)

    num_states = T.data.shape[0]
    num_actions = T.data.shape[1]
    
    Qs = np.zeros((TIMESTEPS, num_states, num_actions)) if Qs is None else Qs
    for i in reversed(range(TIMESTEPS)):
        Q_cur = Qs[i,:]
        Q_next = None
        if i + 1 < TIMESTEPS:
            Q_next = Qs[i+1,:]

        for s in range(num_states):
            for j in range(max_steps_per_state):
                # if j % 500 == 0:
                #     print(f"[Timestep: {i}, State: {s}] Iter {j} of {max_steps_per_state}...")
                eps = 1. - float(j) / max_steps_per_state 
                a = eps_greedy_action(Q_cur, s, eps)
                sp, r = step(T, R, s, a)

                observe(Q_cur, s, a, sp, r, Q_next=Q_next)

    # With Q^* calculated, calculate V^* to compare
    policy = np.zeros((TIMESTEPS, num_states))
    for t in range(TIMESTEPS):
        Q_cur = Qs[t,:]
        for s in range(num_states):
            policy[t,s] = greedy_action(Q_cur, s)

    return Qs, policy.T


def calculate_value(T, R, V, s, pi, V_next=None):
    a = int(pi[s])
    state_val = 0
    num_states = T.data.shape[0]
    for sp in range(num_states):
        next_state_reward = 0
        if V_next is not None:
            next_state_reward = GAMMA * V_next[sp]
        state_val += T(s,a,sp) * (R(s,a,sp) + next_state_reward)

    return state_val


def policy_eval(policy):
    _, T, R, _ = load_data("advanced")

    num_states = T.data.shape[0]
    num_actions = T.data.shape[1]

    Vs = np.zeros((TIMESTEPS, num_states))

    policy = policy.T
    for t in reversed(range(TIMESTEPS)):
        policy_cur = policy[t,:]
        V_cur = Vs[t,:]
        V_next = None
        if t + 1 < TIMESTEPS:
            V_next = Vs[t+1,:]

        for s in range(num_states):
            V_cur[s] = calculate_value(T, R, V_cur, s, policy_cur, V_next=V_next)

    return Vs.T


def compare_policy_eval(policy):
    V_star = np.loadtxt("./policies/value-iter/advanced.v-star")
    Vs = policy_eval(policy)
    rel_err_vec = abs((V_star[:,0] - Vs[:,0]) / V_star[:,0])
    return np.linalg.norm(rel_err_vec)


INITIAL_TRAINING_TIMESTEPS = 10000
SAMPLES = [10, 50, 100, 500, 1000]
NUM_SAMPLES = 100
def main():
    # First train on "models"
    print("Learning basic policy...")
    Q_basic, _ = qlearn_policy("basic", INITIAL_TRAINING_TIMESTEPS)
    print("Learning bad policy...")
    Q_bad, _ = qlearn_policy("bad", INITIAL_TRAINING_TIMESTEPS)

    basic_err_tot = np.zeros((NUM_SAMPLES, len(SAMPLES)))
    bad_err_tot = np.zeros((NUM_SAMPLES, len(SAMPLES)))
    no_err_tot = np.zeros((NUM_SAMPLES, len(SAMPLES)))


    for k in range(NUM_SAMPLES):
        basic_err = basic_err_tot[k,:]
        bad_err = bad_err_tot[k,:]
        no_err = no_err_tot[k,:]
        print(f"Sampling {k + 1} of {NUM_SAMPLES}...")
        for i, timestep in enumerate(SAMPLES):
            print(f"Computing policies for {timestep} timesteps...")
            print("Basic...")
            _, Q_basic_policy = qlearn_policy("advanced", timestep, Qs=Q_basic)
            basic_err[i] = compare_policy_eval(Q_basic_policy)

            print("Bad...")
            _, Q_bad_policy = qlearn_policy("advanced", timestep, Qs=Q_bad)
            bad_err[i] = compare_policy_eval(Q_bad_policy)

            print("No...")
            _, Q_no_policy = qlearn_policy("advanced", timestep)
            no_err[i] = compare_policy_eval(Q_no_policy)

    basic_err = np.mean(basic_err_tot, axis=0)
    bad_err = np.mean(bad_err_tot, axis=0)
    no_err = np.mean(no_err_tot, axis=0)
    basic_std = np.std(basic_err_tot, axis=0)
    bad_std = np.std(bad_err_tot, axis=0)
    no_std = np.std(no_err_tot, axis=0)

    print(f"Basic Error Shape: {basic_err.shape}")

    assert basic_err.shape[0] == len(SAMPLES)
    print("Done computing policy error")
    print("\n\n")

    print("Relative Error L2 Table")
    for sample in SAMPLES:
        print(f"{sample} Steps\t\t", end="")

    print()
    print("Basic Error")
    for v in basic_err:
        print(f"{v}\t", end="")
    print("\nBasic Std Dev")
    for v in basic_std:
        print(f"{v}\t", end="")

    print("\nBad Error")
    for v in bad_err:
        print(f"{v}\t", end="")
    print("\nBad Std Dev")
    for v in bad_std:
        print(f"{v}\t", end="")

    print("\nNo Error")
    for v in no_err:
        print(f"{v}\t", end="")
    print("\nNo Std Dev")
    for v in no_std:
        print(f"{v}\t", end="")
    print()


if __name__ == '__main__':
    main()
