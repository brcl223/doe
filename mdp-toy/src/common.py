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
        self.base_path = f"./transition/{name}"
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


# ns = Number of states
# ts = Number of timesteps in finite horizon
def simulate_run(T, R, policy, ns, ts, ty="closed"):
    state = 0
    reward = 0
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
    return reward
