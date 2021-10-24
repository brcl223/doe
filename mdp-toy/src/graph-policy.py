import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DATA = {
    'advanced': {
        'name': 'advanced',
        'ty': 'value-iter',
        'dispname': 'Advanced (Model-Based)',
    },
    'basic': {
        'name': 'basic',
        'ty': 'value-iter',
        'dispname': 'Basic (Model-Based)'
    },
    'ql': {
        'name': 'advanced',
        'ty': 'q-learning',
        'dispname': 'Q-Learning (Model-Free)'
    },
}

NUM_STATES = 4
NUM_ACTIONS = 4
TIMESTEPS = 10
CHECKPOINT = 2

PALETTE = np.array([[50, 0, 150],
                    [100, 0, 100],
                    [150, 0, 50],
                    [200, 0, 0]])

def load_data(name, ty):
    fpath = f"./policies/{ty}/"
    policy = np.loadtxt(f"{fpath}/{name}.policy")
    V = np.loadtxt(f"{fpath}/{name}-normalized.v-star")

    print(f"Loaded {name}/{ty}")
    print(f"Policy Shape: {policy.shape}")
    print(f"Vs Shape: {V.shape}")
    
    return policy, V


def main():
    # Plot data
    fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(14, 10))

    for i, k in enumerate(DATA):
        v = DATA[k]
        policy, Vs = load_data(v['name'], v['ty'])

        policy = policy[:,1::2]
        Vs = Vs[:,1::2]

        ax1 = axs[0, i]
        ax2 = axs[1, i]
        
        im_policy = ax1.imshow(PALETTE[policy.astype(int)])
        im_value = ax2.imshow(Vs)
        cbar = ax2.figure.colorbar(im_value, ax=ax2, fraction=0.035, pad=0.06)
        im_value.set_clim(0,1)

        ax1.set_ylabel("State")
        ax1.set_xlabel("Timestep")
        ax2.set_ylabel("State")
        ax2.set_xlabel("Timestep")

        timesteps = [str(x) for x in range(2,11,2)]
        ax1.set_xticks(np.arange(len(timesteps)))
        ax2.set_xticks(np.arange(len(timesteps)))
        ax1.set_yticks(np.arange(NUM_ACTIONS))
        ax2.set_yticks(np.arange(NUM_ACTIONS))

        states = [r"$w_1$", r"$w_2$", r"$w_3$", r"$b$"]
        actions = [r"$c_1$", r"$c_2$", r"$c_3$", r"$r$"]
        ax1.set_yticklabels(states)
        ax2.set_yticklabels(states)
        ax1.set_xticklabels(timesteps)
        ax2.set_xticklabels(timesteps)

        ax1.set_title(f"{v['dispname']}\nPolicy per Timestep")
        ax2.set_title(f"State Value per Timestep (Normalized)")

        for i in range(NUM_STATES):
            for j in range(TIMESTEPS//CHECKPOINT):
                ax1.text(j, i, actions[int(policy[i,j])], ha="center", va="center", color="w", fontsize=16)

    # fig.suptitle(f"Model: {NAME}")
    fig.tight_layout()
    plt.savefig(f"./graphs/new/policy-comparison.png")


if __name__ == '__main__':
    main()
