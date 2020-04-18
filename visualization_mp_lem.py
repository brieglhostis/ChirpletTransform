from ChirpletTransform.Chirplet import Chirplet
from ChirpletTransform.AdaptiveChirpletTransform import ACT
import numpy as np
import matplotlib.pyplot as plt


def get_best_I(f, sampling_rate, act_result, n=5):
    if n > len(act_result):
        return act_result
    best_I = np.zeros((n, 4))
    scores = np.zeros(n)
    for I in act_result:
        tc, fc, c, dt_ = I
        chirplet = Chirplet(tc=tc, fc=fc, c=c, dt=dt_, length=len(f), sampling_rate=sampling_rate)
        score = chirplet.chirplet_transform(f)
        i = 0
        while i < n and score < scores[i]:
            i += 1
        if i < n:
            new_best_I = np.copy(best_I)
            new_scores = np.copy(scores)
            new_best_I[i] = I
            new_scores[i] = score
            for j in range(i+1, n):
                new_best_I[j] = best_I[j-1]
                new_scores[j] = scores[j-1]
            best_I = np.copy(new_best_I)
            scores = np.copy(new_scores)
    return best_I


N = 500
sampling_rate = 500
T = N/sampling_rate
dt = 1

chirplet1 = Chirplet(tc=0.2, fc=43, c=142, dt=dt/5, length=N, sampling_rate=sampling_rate, gaussian=True)
chirplet2 = Chirplet(tc=0.7, fc=74, c=-34, dt=2*T, length=N, sampling_rate=sampling_rate, gaussian=False)

signal = np.real(chirplet1.chirplet + 1*chirplet2.chirplet)
plt.plot(np.arange(N)/sampling_rate, signal, label="Target signal")
plt.title("Target signal")
plt.legend()
plt.show()

act = ACT(signal, sampling_rate=sampling_rate, P=100)
act.plot_wigner_distribution(title="Target signal")

act_mp = act.act_gradient_descent(tc=T/2, dt=dt/5, fixed_tc=False)

approximation_mp = np.zeros(N)

for I in get_best_I(signal, sampling_rate, act_mp, n=10):
    tc, fc, c, dt_ = I
    chirplet = Chirplet(tc=tc, fc=fc, c=c, dt=dt_, length=N, sampling_rate=sampling_rate)
    print(I, chirplet.chirplet_transform(signal))
    approximation_mp += chirplet.chirplet_transform(signal)*np.real(chirplet.chirplet)

plt.plot(np.arange(N)/sampling_rate, approximation_mp, label="MP approximation")
plt.title("MP approximation")
plt.legend()
plt.show()

ACT(approximation_mp, sampling_rate=sampling_rate).plot_wigner_distribution(title="MP algorithm")

approximation_mp_lem = np.zeros(N)

act_mp_lem = act.act_mp_lem(tc=T/2, dt=dt/5)

for I in get_best_I(signal, sampling_rate, act_mp_lem, n=100):
    tc, fc, c, dt_ = I
    chirplet = Chirplet(tc=tc, fc=fc, c=c, dt=dt_, length=N, sampling_rate=sampling_rate)
    print(I, chirplet.chirplet_transform(signal))
    approximation_mp_lem += chirplet.chirplet_transform(signal)*np.real(chirplet.chirplet)

plt.plot(np.arange(N)/sampling_rate, approximation_mp_lem, label="MP-LEM approximation")
plt.title("MP-LEM approximation")
plt.legend()
plt.show()

ACT(approximation_mp_lem, sampling_rate=sampling_rate).plot_wigner_distribution(title="MP-LEM algorithm")


