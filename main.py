from Chirplet import Chirplet
from AdaptiveChirpletTransform import ACT
import numpy as np
import matplotlib.pyplot as plt


def test():
    N = 400
    sampling_rate = 100
    I = (1.5, 10, 2, 2)
    chirplet = Chirplet(tc=I[0], fc=I[1], c=I[2], dt=I[3], length=N, sampling_rate=sampling_rate, gaussian=True)
    print(chirplet.get_starting_and_ending_frequencies())
    chirplet.plot_chirplet()
    chirplet.plot_wigner_distribution()

    act = ACT(chirplet.chirplet, sampling_rate=sampling_rate, P=1)
    print(act.act_greedy(tc=1.5, dt=2))
    act.plot_wigner_distribution()

if __name__=="__main__":
    test()
