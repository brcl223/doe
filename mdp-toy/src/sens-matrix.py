import configparser as cfg
import csv
import glob
import os
import random
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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


def main():
    closed_policies = load_policies()
    open_policies = load_policies(ty="open")
    row_len = len(closed_policies)
    closed_rewards = np.zeros((row_len, row_len))
    open_rewards = np.zeros((row_len, row_len))
    semi_rewards = np.zeros((row_len, row_len))
    for i, policy in enumerate(closed_policies):
        name = policy[0]
        print(f"Current evaluating policies for {name}...")
        c_data, s_data, o_data = simulate_row(name, closed_policies, open_policies)
        all_data = np.concatenate((c_data, s_data, o_data))
        all_data_norm = normalize_row(all_data)
        closed_rewards[i,:] = all_data_norm[0:row_len]
        semi_rewards[i,:] = all_data[row_len:row_len*2]
        open_rewards[i,:] = all_data[row_len*2:row_len*3]

    if SAVE_RUN:
        base_name = "./data/rewards/"
        np.savetxt(f"{base_name}/closed_rewards.txt", closed_rewards)
        np.savetxt(f"{base_name}/semi_rewards.txt", semi_rewards)
        np.savetxt(f"{base_name}/open_rewards.txt", open_rewards)

    # rewards = np.delete(rewards,np.s_[3:5],axis=0)
    # rewards = np.delete(rewards,np.s_[3:5],axis=1)
    # Plotting reward matrix
    fig, (axc, axs, axo) = plt.subplots(ncols=3)
    imc = axc.imshow(closed_rewards)
    axc.figure.colorbar(imc, ax=axc)
    imo = axo.imshow(open_rewards)
    axo.figure.colorbar(imo, ax=axo)
    ims = axs.imshow(semi_rewards)
    axs.figure.colorbar(ims, ax=axs)

    pnames = ["Aero",
              "Add.",
              "Auto",
              "Agr.",
              "Comp",
              "Defense",
              "Loco",
              "Marine",
              "Power",]

    axc.set_title("CL")
    axo.set_title("OL")
    axs.set_title("Semi-OL")
    fig.suptitle("Sensitivity Matrix")
    # # row_len = len(rewards)
    # ax.set_ylabel("Manufacturing Process")
    # ax.set_xlabel("Policy")
    # ax.set_xticks(np.arange(row_len))
    # ax.set_yticks(np.arange(row_len))
    # # pnames = [p[0] for p in policies]
    # ax.set_yticklabels(pnames, rotation=45)
    # ax.set_xticklabels(pnames, rotation=45)
    # ax.set_title("Reduced Policy Sensitivity Matrix")

    # for i in plt.gca().get_xticklabels():
    #     if i.get_position()[0] in [4,5]:
    #         i.set_color("red")
    #     if i.get_position()[0] in [3,6]:
    #         i.set_color("#151351")

    # for i in plt.gca().get_yticklabels():
    #     if i.get_position()[1] in [4,5]:
    #         i.set_color("red")
    #     if i.get_position()[1] in [3,6]:
    #         i.set_color("#151351")

    plt.show()


def simulate_row(name, clp, olp):
    cfg, T, R, _ = load_data(name)
    ns = int(cfg.fields["MDP"]["states"])
    ts = int(cfg.fields["MDP"]["timesteps"])
    print("Beginning CL evaluation...")
    clr = simulate_row_closed(name, clp, T, R, ns, ts)
    print("Beginning OL evaluation...")
    olr, slr = simulate_row_open(name, olp, T, R, ns, ts)
    print("Evaluation finished")
    return clr, slr, olr
    

def simulate_row_closed(name, policies, T, R, num_states, timesteps):
    rewards = []
    for (_, policy) in policies:
        rewards.append(0)
        for run in range(NUM_RUNS):
            if (run + 1) % 100 == 0:
                print(f"Current closed loop iter: {run + 1}")
            rewards[-1] += simulate_run_closed(T, R, policy, num_states, timesteps)
        rewards[-1] /= float(NUM_RUNS)
    return np.array(rewards)


def simulate_row_open(name, policies, T, R, num_states, timesteps):
    open_rewards = []
    semi_rewards = []
    num_policies = len(policies) // num_states
    for i in range(num_policies):
    # for (_, policy) in policies:
        cur_policies = list(map(lambda x: x[1], policies[i:i+num_states]))
        open_rewards.append(0)
        semi_rewards.append(0)
        for run in range(NUM_RUNS):
            if (run + 1) % 100 == 0:
                print(f"Current open loop iter: {run + 1}")
            open_rewards[-1] += simulate_run_open(T, R, cur_policies[0], num_states, timesteps)
            semi_rewards[-1] += simulate_run_semi(T, R, cur_policies, num_states, timesteps)
        open_rewards[-1] /= float(NUM_RUNS)
        semi_rewards[-1] /= float(NUM_RUNS)
    return np.array(open_rewards), np.array(semi_rewards)


def normalize_row(row_data):
    if SUM_NORMALIZE:
        min_val = row_data.min()
        row_data -= min_val
        max_val = row_data.max()
        row_data /= max_val
    else:
        norm_val = max(abs(row_data.min()), abs(row_data.max()))
        row_data /= norm_val

    return row_data


def load_policies(ty="closed"):
    policies = []
    if not (ty == "closed" or ty == "open"):
        raise RuntimeError(f"Invalid policy type: {ty}\nMust be 'open' or 'closed'")
    
    for policy in glob.glob(f"./policies/{ty}-loop/*.policy"):
        name = re.search(fr"\./policies/{ty}-loop/(.*)\.policy", policy)
        p = (name[1], np.loadtxt(policy))
        policies.append(p)

    policies.sort(key=lambda x: x[0])
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
def simulate_run_closed(T, R, policy, ns, ts):
    reward = 0
    state = 0
    for _ in range(NUM_RUNS):
        for t in range(ts):
            action = int(policy[state,t])
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


def simulate_run_open(T, R, policy, ns, ts):
    REPLACE_ACTION = 3
    INITIAL_STATE = 0
    reward = 0
    for j in range(NUM_RUNS):
        if j != 0:
            reward += R(state,REPLACE_ACTION,INITIAL_STATE)
        state = INITIAL_STATE
        for t in range(ts):
            action = int(policy[t])
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


def simulate_run_semi(T, R, policies, ns, ts):
    reward = 0
    state = 0
    for _ in range(NUM_RUNS):
        policy = policies[state]
        for t in range(ts):
            action = int(policy[t])
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


# Set to true for low-loop count test run
# Set to false for real run
TEST_MODE = False

# Number runs per policy for monte-carlo simulation
if TEST_MODE:
    NUM_RUNS = 5
    print("##########################")
    print("Warning: Running in TEST MODE")
    print("##########################")
else:
    NUM_RUNS = 1000
# If not summation normalization, then multiplicative normalization
# Summation is [0,1], Multiplicative is [-1,1]
SUM_NORMALIZE = True
SAVE_RUN = True

if __name__ == '__main__':
    main()
