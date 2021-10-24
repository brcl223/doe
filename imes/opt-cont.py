import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import imes as model


def cutspeed_cost(vb, vc, fpt, alpha=0):
    Jvc = (vc - min(model.thermal_damage_limit(vb, fpt), 90)) ** 2
    Ju = -vc
    return alpha * Ju + Jvc


def fpt_cost(vb, fpt, alpha=0):
    Jfr = (fpt - model.feed_rate_damage_limit(vb)) ** 2
    Ju = -fpt
    return alpha * Ju + Jfr


def simulate(cut_length=53, alpha=0, beta=0):
    VB_MAX = 50

    vc_range = np.linspace(1, 90, num=(90))
    fpt_range = np.linspace(0.001, 0.05, num=151)

    def f(t, y, vc, fpt):
        L = y[0]
        return model.wdot(L, vc, fpt)

    fpts = []
    vcs = []
    Ls = [ 0.01 ]
    vbs = [ 0.01 ]
    times = [ 0. ]

    L = 0.

    # Debugging
    frs = [0.]
    cts = [0.]

    while vbs[-1] < VB_MAX:
        L_next = None
        vb_next = None
        min_cost = float('inf')
        t0_next = None
        vc_next = None

        L = Ls[-1]
        vb = vbs[-1]
        t0 = times[-1]

        fr_next = None
        cut_time_next = None
        fpt_max_idx = fpt_cost(vb, fpt_range, alpha=beta).argmin()
        fpt_max = fpt_range[fpt_max_idx]

        for vc in vc_range:
            fr = model.feed_rate(vc, fpt_max)
            cut_time = cut_length / fr
            sol = solve_ivp(f, [t0, t0 + cut_time], [L, vb], args=(vc, fpt_max))
            L_tmp = sol.y[0][-1]
            vb_tmp = sol.y[1][-1]
            cur_cost = cutspeed_cost(vb_tmp, vc, fpt_max, alpha=alpha)

            if cur_cost < min_cost:
                fr_next = fr
                cut_time_next = cut_time

                min_cost = cur_cost
                L_next = L_tmp
                vb_next = vb_tmp
                vc_next = vc
                t0_next = t0 + cut_time

        frs.append(fr_next)
        cts.append(cut_time_next)
        times.append(t0_next)
        Ls.append(L_next)
        vbs.append(vb_next)
        fpts.append(fpt_max)
        vcs.append(vc_next)

    vc_avg = np.average(vcs)
    fpt_avg = np.average(fpts)
    fr_avg = np.average(frs)

    stats = {
        'vc_avg': vc_avg,
        'fpt_avg': fpt_avg,
        'fr_avg': fr_avg,
        'slide_distance': L,
    }

    return np.array(times), np.array(Ls), np.array(vbs), np.array(vcs), np.array(fpts), stats



def plot(T, L, Vb, Vc, Fpts, fname):
    vbs = np.linspace(0.5, 50, num=100)
    vc_limit = []
    i = 0
    Vb = Vb[1:]

    for vb in vbs:
        if vb > Vb[i]:
            i = min(i+1, len(Fpts)-2)
        fptcur = Fpts[i]
        vc_limit.append(model.thermal_damage_limit(vb, fptcur))

    assert len(vc_limit) == len(vbs)
    fpts_limit = model.feed_rate_damage_limit(vbs)

    fig, (ax1, ax2) = plt.subplots(nrows=2)

    ax1.step(Vb, Vc, label="Input")
    ax1.plot(vbs, vc_limit, 'k--', label="Damage Limit")
    ax1.set_xlabel("VB (µm)")
    ax1.set_ylabel("$V_c$ (m/min)")
    ax1.set_ylim([0,200])
    ax1.set_xlim([0,50])
    ax1.set_title("Cutspeed Control")
    ax1.legend(loc="upper right")

    ax2.step(Vb, Fpts, label="Input")
    ax2.plot(vbs, fpts_limit, 'k--', label="Damage Limit")
    ax2.set_xlabel("VB (µm)")
    ax2.set_ylabel("Feed/Tooth (mm/tooth/rev)")
    ax2.set_title("Feed Rate Control")
    ax2.legend(loc="upper right")
    ax2.set_xlim([0,50])

    fig.suptitle("Optimal Control Strategy")
    fig.tight_layout()
    fig.savefig(f"./plots/{fname}.png")


if __name__ == '__main__':
    T, L, Vb, Vc, Fpts, stats = simulate(cut_length=53, alpha=0, beta=0)
    plot(T, L, Vb, Vc, Fpts, "opt-cont-discrete-cuts-updated")

    print(f"Profitability: {model.calc_profit(stats)}")
    print(f"Energy: {model.calc_energy(stats)}")
