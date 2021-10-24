import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


NUM_STATES = 4
NUM_ACTIONS = 4
TIMESTEPS = 10
CHECKPOINT = 2

NAMES = ["advanced-value-iter", "basic-value-iter", "advanced-q-learning"]
TITLE_NAMES = ["Basic", "Q-Learning"]

def load_data(name):
    fpath = f"./policies/robustness/"
    return np.loadtxt(f"{fpath}/advanced-vs-{name}.v-star")


def load_data_and_compute_rel_error():
    adv = load_data(NAMES[0])
    basic = load_data(NAMES[1])
    qlr = load_data(NAMES[2])

    assert adv.shape == basic.shape == qlr.shape

    basic = abs((basic - adv) / adv)
    qlr = abs((qlr - adv) / adv)

    return basic, qlr

def main():
    # Plot data
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(8,12))

    data = load_data_and_compute_rel_error()
    max_val = 0
    for V in data:
        max_val = max(max_val, V.max())
    
    for i, V in enumerate(data):
        Vs = V[:,1::2]

        ax = axs[i]
        
        # im = ax.imshow(Vs)
        im = ax.imshow(Vs)
        # ax.figure.colorbar(im, ax=ax)
        im.set_clim(0,0.5)

        ax.set_ylabel("State")
        ax.set_xlabel("Timestep")

        timesteps = [str(x) for x in range(2,11,2)]
        ax.set_xticks(np.arange(len(timesteps)))
        ax.set_yticks(np.arange(NUM_ACTIONS))

        states = [r"$w_{1}$", r"$w_{2}$", r"$w_3$", r"$b$"]
        ax.set_yticklabels(states)
        ax.set_xticklabels(timesteps)

        ax.set_title(f"{TITLE_NAMES[i]}")

    fig.colorbar(im, ax=axs.ravel().tolist())
    # fig.suptitle(f"")
    # fig.tight_layout()
    fig.savefig("./graphs/new/robust-v-clamped.png")
    # plt.savefig(f"./graphs/new/robust-v.png")


if __name__ == '__main__':
    main()
