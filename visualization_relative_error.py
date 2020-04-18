import numpy as np
import matplotlib.pyplot as plt

from ChirpletTransform.Chirplet import Chirplet
from ChirpletTransform.AdaptiveChirpletTransform import ACT


def load_data_set():
    data_set = np.load("Chirp_data_set/dataset.npy")

    (a, b) = np.shape(data_set)
    M = int(np.round(np.sqrt(a)))
    N = b - 2
    sampling_rate = 500
    max_fqcy = sampling_rate/2
    c_range = np.arange(M) * max_fqcy / (M*N/sampling_rate)
    fc_range = np.arange(M) * max_fqcy / M

    return data_set, c_range, fc_range, M, N


def get_best_chirplet(f, I_list, length=500, sampling_rate=500):
    best_I = I_list[0]
    best_sc = 0
    for I in I_list:
        tc, fc, c, dt = I
        chirplet = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=length, sampling_rate=sampling_rate)
        sc = chirplet.chirplet_transform(f)
        if sc > best_sc:
            best_I = I
            best_sc = sc
    return best_I


if __name__ == "__main__":

    data_set, c_range, fc_range, M, N = load_data_set()
    dt = 1
    sampling_rate = N
    print(M, N)

    error_greedy = np.zeros((M, M))
    error_steve = np.zeros((M, M))
    error_gd_fixed_tc = np.zeros((M, M))
    error_gd = np.zeros((M, M))

    m = 0
    for row in data_set:

        act = ACT(row[:-2], sampling_rate=1000, P=5)
        c_m = row[-2]
        f_m = row[-1]

        # Greedy algorithm
        act_greedy = act.act_greedy(tc=dt/2, dt=dt)
        I_greedy = get_best_chirplet(row[:-2], act_greedy, length=N, sampling_rate=sampling_rate)
        c_m_pred = np.abs(I_greedy[2])
        error = np.abs(c_m - c_m_pred)
        if c_m == 0:
            error_greedy[m % M][m // M] = error / c_range[1]
        else:
            error_greedy[m % M][m // M] = error / c_m  # rows: c cst, col: w0 cst
        print(m, "Greedy alg, label: ", c_m, ", prediction: ", c_m_pred, ", error: ", error)

        # Steve's algorithm
        act_steve = act.act_steve_mann(dt=dt)
        I_steve = get_best_chirplet(row[:-2], act_steve, length=N, sampling_rate=sampling_rate)
        c_m_pred = np.abs(I_steve[2])
        error = np.abs(c_m - c_m_pred)
        if c_m == 0:
            error_steve[m % M][m // M] = error / c_range[1]
        else:
            error_steve[m % M][m // M] = error / c_m  # rows: c cst, col: w0 cst
        print(m, "Steve's alg, label: ", c_m, ", prediction: ", c_m_pred, ", error: ", error)

        # Gradient descent algorithm fixed tc
        act_gd_fixed_tc = act.act_gradient_descent(tc=dt / 2, dt=dt, fixed_tc=True)
        I_gd_fixed_tc = get_best_chirplet(row[:-2], act_gd_fixed_tc, length=N, sampling_rate=sampling_rate)
        c_m_pred = np.abs(I_gd_fixed_tc[2])
        error = np.abs(c_m - c_m_pred)
        if c_m == 0:
            error_gd_fixed_tc[m % M][m // M] = error / c_range[1]
        else:
            error_gd_fixed_tc[m % M][m // M] = error / c_m  # rows: c cst, col: w0 cst
        print(m, "Gradient descent 1, label: ", c_m, ", prediction: ", c_m_pred, ", error: ", error)

        # Gradient descent algorithm
        act_gd = act.act_gradient_descent(tc=dt / 2, dt=dt, fixed_tc=False)
        I_gd = get_best_chirplet(row[:-2], act_gd, length=N, sampling_rate=sampling_rate)
        c_m_pred = np.abs(I_gd[2])
        error = np.abs(c_m - c_m_pred)
        if c_m == 0:
            error_gd[m % M][m // M] = error / c_range[1]
        else:
            error_gd[m % M][m // M] = error / c_m  # rows: c cst, col: w0 cst
        print(m, "Gradient descent 2, label: ", c_m, ", prediction: ", c_m_pred, ", error: ", error)

        m += 1

    plt.plot(c_range, np.mean(error_greedy, axis=1), label="Greedy algorithm")
    plt.plot(c_range, np.mean(error_steve, axis=1), label="Dichotomy algorithm")
    plt.plot(c_range, np.mean(error_gd_fixed_tc, axis=1), label="Gradient descent algorithm, fixed tc")
    plt.plot(c_range, np.mean(error_gd, axis=1), label="Gradient descent algorithm")
    plt.xlabel("Chirpiness")
    plt.ylabel("Relative error")
    plt.legend()
    plt.show()

    """
    error_fc = np.mean(error_steve, axis=0)
    error_fc_std = np.std(error_steve, axis=0)

    plt.plot(fc_range, error_fc, label="Mean error")
    plt.plot(fc_range, error_fc + error_fc_std, label="Standard deviation upper bound")
    plt.plot(fc_range, error_fc - error_fc_std, label="Standard deviation lower bound")
    plt.xlabel("Starting frequency")
    plt.ylabel("Relative error")
    plt.legend()
    # plt.savefig("error_vs_w0_N" + str(N) + "_M" + str(M) + ".jpg")
    plt.show()
    """
