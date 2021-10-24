import numpy as np
import configparser as cfg
import csv
import os
import sys
import random

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
        self.base_path = f"./transitions/{name}"
        self.fields = load_config(self.base_path)


### Data saving/loading utilities
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


# ns = Number of states
# ts = Number of timesteps in finite horizon
def simulate_run(T, R, policy, ns, ts, ty="closed"):
    reward = 0
    for _ in range(NUM_RUNS):
        state = START_STATE
        for t in range(ts):
            if ty == "closed":
                action = int(policy[state,t])
            elif ty == "open":
                action = int(policy[t])
            else:
                raise "Invalid simulation type. ty=(closed|open)"
            state_val = random.random()
            state_prob = 0
            for i in range(ns):
                state_prob += T(state,action,i)
                assert state_prob <= 1.05, f"State probability range beyond 1.\nValue: {state_prob}"
                if state_val <= state_prob:
                    reward += R(state,action,i)
                    state = i
                    break
    return reward / NUM_RUNS







def main():
    print(f"Beginning simulation for {NAME}...")
    cfg, T, R, _ = load_data(NAME)
    ns = int(cfg.fields["MDP"]["states"])
    na = int(cfg.fields["MDP"]["actions"])
    ts = int(cfg.fields["MDP"]["timesteps"])
    policy = np.zeros(ts)
    max_reward = float("-inf")
    best_policy = None

    TOTAL_ITERS = na**ts
    for i in range(TOTAL_ITERS):
        if i % 2000 == 0:
            print(f"[{NAME}]: Current iteration: {i} of {TOTAL_ITERS}")
        cur_reward = simulate_run(T,R,policy,ns,ts,ty="open")
        if cur_reward > max_reward:
            max_reward = cur_reward
            best_policy = policy.copy()
        policy = update_policy(policy,na)

    print("Optimal Policy:")
    print(best_policy)
    print(f"\nMax Reward: {max_reward}")
    print(f"Finished simulation for {NAME}...")

    if SAVE_RUN:
        print("Saving results...")
        fname = f"./policies/open-loop/{NAME}.state-{START_STATE}.policy"
        np.savetxt(fname, best_policy, header=f"Max Reward: {max_reward}")
        print("Results saved")

    
# na = Number of Actions
def update_policy(policy, na):
    for i in range(policy.shape[0]):
        policy[i] += 1
        if policy[i] >= na:
            policy[i] = 0
        else:
            break
    return policy

NAME = sys.argv[1]
START_STATE = int(sys.argv[2])
NUM_RUNS = 500
SAVE_RUN = True

if __name__ == '__main__':
    main()
