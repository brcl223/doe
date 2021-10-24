import numpy as np


BASE_PATH = "./policies/"


def main():
    v_adv = np.loadtxt(f"{BASE_PATH}/value-iter/advanced.v-star")
    v_basic = np.loadtxt(f"{BASE_PATH}/value-iter/basic.v-star")
    v_ql = np.loadtxt(f"{BASE_PATH}/q-learning/advanced.v-star")

    min_val = min(v_adv.min(), v_basic.min(), v_ql.min())
    max_val = max(v_adv.max(), v_basic.max(), v_ql.max())

    for v in (v_adv, v_basic, v_ql):
        v -= min_val
        v /= abs(max_val - min_val)

    np.savetxt(f"{BASE_PATH}/value-iter/advanced-normalized.v-star", v_adv)
    np.savetxt(f"{BASE_PATH}/value-iter/basic-normalized.v-star", v_basic)
    np.savetxt(f"{BASE_PATH}/q-learning/advanced-normalized.v-star", v_ql)


if __name__ == '__main__':
    main()
