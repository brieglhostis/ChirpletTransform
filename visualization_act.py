from ChirpletTransform.Chirplet import Chirplet
from ChirpletTransform.AdaptiveChirpletTransform import ACT
import numpy as np
import matplotlib.pyplot as plt


N = 1000
sampling_rate = 1000
dt = 1

chirp = Chirplet(tc=0.3, fc=22.7, c=33.2, dt=0.4 + N/sampling_rate, length=N, sampling_rate=sampling_rate, gaussian=False)
chirp.plot_chirplet(label="Target chirp", title="Target chirp, tc = 0.3, fc = 22.7, c = 33.2")
#chirp.plot_wigner_distribution()

act = ACT(chirp.chirplet, sampling_rate=sampling_rate, P=1)
#act.plot_wigner_distribution(title="Target chirp, tc = 0.3, fc = 22.7, c = 33.2")

alg = 4

# Greedy act
if alg == 0:
    act_res = act.act_greedy(tc=0.3, dt=dt)
elif alg == 1:
    act_res = act.act_exhaustive(tc=0.3, dt=dt)
elif alg == 2:
    act_res = act.act_gradient_descent(tc=0.3, dt=dt, fixed_tc=True)
elif alg == 3:
    act_res = act.act_gradient_descent(tc=0.5, dt=dt, fixed_tc=False)
else:
    act_res = act.act_steve_mann(t0=0, dt=dt)

approximation = np.zeros(N)
for I in act_res:
    tc, fc, c, dt = I
    chirplet = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=N, sampling_rate=sampling_rate)
    print(I, chirplet.chirplet_transform(chirp.chirplet))
    approximation += chirplet.chirplet_transform(chirp.chirplet)*np.real(chirplet.chirplet)

ACT(approximation, sampling_rate=sampling_rate).plot_wigner_distribution(title=
                                                                             "Gradient descent")

