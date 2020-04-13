from Chirplet import Chirplet
from AdaptiveChirpletTransform import ACT
import numpy as np
import matplotlib.pyplot as plt


def swiping_fixed_chirpiness():
    N = 2000
    sampling_rate = 100
    dt = 1

    t = np.arange(N)/sampling_rate
    frequency = 0.25*sampling_rate*(1+np.cos(2*np.pi*0.1*t))

    signal = np.cos(2*np.pi*frequency*t)

    plt.plot(t, frequency)
    plt.plot(t, signal)
    plt.show()

    chirpiness_number = 200  # number of chirpiness (excluding 0) to try on the signal
    c_range = (np.arange(chirpiness_number+1)/chirpiness_number - 0.5)*2*sampling_rate/dt  # Chripiness ranges from -sampling_rate/(dt) to sampling_rate/(dt)
    print("Chirpiness range:", c_range)

    number_of_estimations = 200
    c_test = np.zeros((chirpiness_number+1, number_of_estimations))
    c_true = np.zeros(number_of_estimations)

    for index in range(number_of_estimations):

        i0 = int(index*(N - dt * sampling_rate) / number_of_estimations)
        windowed_signal = signal[i0: i0 + int(dt * sampling_rate)]
        act = ACT(windowed_signal, sampling_rate=sampling_rate)

        for i in range(chirpiness_number+1):
            c = c_range[i]
            fc = act.best_fitting_frequency(dt/2, c)
            chirplet = Chirplet(tc=dt/2, fc=fc, c=c, dt=dt, length=sampling_rate*dt, sampling_rate=sampling_rate,
                                gaussian=False)
            c_test[i][index] = chirplet.chirplet_transform(windowed_signal)

        c_true[index] = (frequency[int(i0 + sampling_rate * dt)] - frequency[i0]) / dt

    for i in range(chirpiness_number):
        plt.plot(np.arange(number_of_estimations)*N/(sampling_rate*number_of_estimations), c_range[i] + 10*c_test[i])
    plt.plot(np.arange(number_of_estimations)*N/(sampling_rate*number_of_estimations), c_true, label="True chirpiness")
    plt.legend()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(np.arange(number_of_estimations) * N / (sampling_rate * number_of_estimations), c_true,
             label="True chirpiness")
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Chirpiness [Hz^2]')
    ax2.imshow(np.flip(c_test, axis=0), extent=(0, N/sampling_rate, c_range[0], c_range[-1]), aspect=N*dt/(8*sampling_rate**2))
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Chirpiness [Hz^2]')
    plt.show()


def swiping_fixed_frequency():
    N = 2000
    sampling_rate = 100
    dt = 1

    t = np.arange(N)/sampling_rate
    frequency = 0.25*sampling_rate*(1+np.cos(2*np.pi*0.1*t))

    signal = np.cos(2*np.pi*frequency*t)

    plt.plot(t, frequency)
    plt.plot(t, signal)
    plt.show()

    frequency_number = 200  # number of frequencies to try on the signal
    fc_range = (np.arange(frequency_number+1)/frequency_number)*sampling_rate  # Frequency ranges from 0 to sampling_rate
    print("Range of frequencies:", fc_range)

    number_of_estimations = 200
    fc_test = np.zeros((frequency_number+1, number_of_estimations))
    fc_true = np.zeros(number_of_estimations)

    for index in range(number_of_estimations):

        i0 = int(index*(N - dt * sampling_rate) / number_of_estimations)
        windowed_signal = signal[i0: i0 + int(dt * sampling_rate)]
        act = ACT(windowed_signal, sampling_rate=sampling_rate)

        for i in range(frequency_number+1):
            fc = fc_range[i]
            c = act.best_fitting_chirpiness(dt/2, fc, dt)
            chirplet = Chirplet(tc=dt/2, fc=fc, c=c, dt=dt, length=sampling_rate*dt, sampling_rate=sampling_rate,
                                gaussian=False)
            fc_test[i][index] = chirplet.chirplet_transform(windowed_signal)

        fc_true[index] = frequency[int(i0 + sampling_rate * dt/2)]

    for i in range(frequency_number+1):
        plt.plot(np.arange(number_of_estimations)*N/(sampling_rate*number_of_estimations), fc_range[i] + 10*fc_test[i])
    plt.plot(np.arange(number_of_estimations)*N/(sampling_rate*number_of_estimations), fc_true, label="True chirpiness")
    plt.legend()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(np.arange(number_of_estimations) * N / (sampling_rate * number_of_estimations), fc_true,
             label="True frequency")
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Frequency [Hz]')
    ax2.imshow(np.flip(fc_test, axis=0), extent=(0, N/sampling_rate, fc_range[0], fc_range[-1]), aspect=N/(4*sampling_rate**2))
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Frequency [Hz]')
    plt.show()


if __name__=="__main__":
    swiping_fixed_chirpiness()
    swiping_fixed_frequency()
