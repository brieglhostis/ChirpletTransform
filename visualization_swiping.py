from ChirpletTransform.Chirplet import Chirplet
from ChirpletTransform.AdaptiveChirpletTransform import ACT
import numpy as np
import matplotlib.pyplot as plt


def swiping_fixed_chirpiness(signal, sampling_rate=1.0, dt=1.0, chirpiness_number=200, number_of_estimations=200):
    # chirpiness number: number of chirpiness (excluding 0) to try on the signal
    N = len(signal)

    c_range = (np.arange(chirpiness_number+1)/chirpiness_number - 0.5)*sampling_rate/dt  # Chripiness ranges from -sampling_rate/(dt) to sampling_rate/(dt)
    print("Chirpiness range:", c_range[0], "to", c_range[-1])

    """
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(nrows=8)
    ax1.plot(np.arange(int(sampling_rate*dt))/sampling_rate, Chirplet(tc=dt / 2, fc=sampling_rate/8, c=c_range[int(0*number_of_estimations/8)], dt=dt, length=int(sampling_rate * dt), sampling_rate=sampling_rate, gaussian=False).chirplet)
    ax2.plot(np.arange(int(sampling_rate*dt))/sampling_rate, Chirplet(tc=dt / 2, fc=sampling_rate/8, c=c_range[int(1*number_of_estimations/8)], dt=dt, length=int(sampling_rate * dt), sampling_rate=sampling_rate, gaussian=False).chirplet)
    ax3.plot(np.arange(int(sampling_rate*dt))/sampling_rate, Chirplet(tc=dt / 2, fc=sampling_rate/8, c=c_range[int(2*number_of_estimations/8)], dt=dt, length=int(sampling_rate * dt), sampling_rate=sampling_rate, gaussian=False).chirplet)
    ax4.plot(np.arange(int(sampling_rate*dt))/sampling_rate, Chirplet(tc=dt / 2, fc=sampling_rate/8, c=c_range[int(3*number_of_estimations/8)], dt=dt, length=int(sampling_rate * dt), sampling_rate=sampling_rate, gaussian=False).chirplet)
    ax5.plot(np.arange(int(sampling_rate*dt))/sampling_rate, Chirplet(tc=dt / 2, fc=sampling_rate/8, c=c_range[int(4*number_of_estimations/8)], dt=dt, length=int(sampling_rate * dt), sampling_rate=sampling_rate, gaussian=False).chirplet)
    ax6.plot(np.arange(int(sampling_rate*dt))/sampling_rate, Chirplet(tc=dt / 2, fc=sampling_rate/8, c=c_range[int(5*number_of_estimations/8)], dt=dt, length=int(sampling_rate * dt), sampling_rate=sampling_rate, gaussian=False).chirplet)
    ax7.plot(np.arange(int(sampling_rate*dt))/sampling_rate, Chirplet(tc=dt / 2, fc=sampling_rate/8, c=c_range[int(6*number_of_estimations/8)], dt=dt, length=int(sampling_rate * dt), sampling_rate=sampling_rate, gaussian=False).chirplet)
    ax8.plot(np.arange(int(sampling_rate*dt))/sampling_rate, Chirplet(tc=dt / 2, fc=sampling_rate/8, c=c_range[int(7*number_of_estimations/8)], dt=dt, length=int(sampling_rate * dt), sampling_rate=sampling_rate, gaussian=False).chirplet)
    plt.show()
    """

    c_test = np.zeros((chirpiness_number+1, number_of_estimations))
    c_true = np.zeros(number_of_estimations)

    for index in range(number_of_estimations):

        i0 = int(index*(N - dt * sampling_rate) / number_of_estimations)
        windowed_signal = signal[i0: i0 + int(dt * sampling_rate)]
        act = ACT(windowed_signal, sampling_rate=sampling_rate)

        for i in range(chirpiness_number+1):
            c = c_range[i]
            fc = act.best_fitting_frequency(dt/2, c)
            chirplet = Chirplet(tc=dt/2, fc=fc, c=c, dt=dt, length=int(sampling_rate*dt), sampling_rate=sampling_rate,
                                gaussian=False)
            c_test[i][index] = chirplet.chirplet_transform(windowed_signal)

        c_true[index] = (frequency[int(i0 + sampling_rate * dt)] - frequency[i0]) / dt

    c_test = c_test + np.flip(c_test, axis=0)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(dt/2 + np.arange(number_of_estimations) * (N-dt*sampling_rate/2)/(sampling_rate * number_of_estimations), c_true,
             label="True chirpiness")
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Chirpiness [Hz^2]')
    ax1.legend()
    ax2.imshow(np.flip(c_test, axis=0), extent=(dt/2, N/sampling_rate - dt/2, c_range[0], c_range[-1]),
               aspect=(N-dt*sampling_rate)*dt/(4*sampling_rate**2))
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Chirpiness [Hz^2]')
    plt.show()


def swiping_fixed_frequency(signal, sampling_rate=1.0, dt=1.0, frequency_number=200, number_of_estimations=200):
    # frequency_number: number of frequencies to try on the signal
    N = len(signal)

    fc_range = (np.arange(frequency_number+1)/frequency_number)*sampling_rate/2  # Frequency ranges from 0 to sampling_rate
    print("Range of frequencies:", fc_range[0], "to", fc_range[-1])

    fc_test = np.zeros((frequency_number+1, number_of_estimations))
    fc_true = np.zeros(number_of_estimations)

    for index in range(number_of_estimations):

        i0 = int(index*(N - dt * sampling_rate) / number_of_estimations)
        windowed_signal = signal[i0: i0 + int(dt * sampling_rate)]
        act = ACT(windowed_signal, sampling_rate=sampling_rate)

        for i in range(frequency_number+1):
            fc = fc_range[i]
            c = act.best_fitting_chirpiness(dt/2, fc, dt)
            chirplet = Chirplet(tc=dt/2, fc=fc, c=c, dt=dt, length=int(sampling_rate*dt), sampling_rate=sampling_rate,
                                gaussian=False)
            fc_test[i][index] = chirplet.chirplet_transform(windowed_signal)

        fc_true[index] = frequency[int(i0 + sampling_rate * dt/2)]

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(dt/2 + np.arange(number_of_estimations) * (N-dt*sampling_rate/2)/(sampling_rate * number_of_estimations), fc_true,
             label="True frequency")
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.legend()
    ax2.imshow(np.flip(fc_test, axis=0), extent=(dt/2, N/sampling_rate - dt/2, fc_range[0], fc_range[-1]),
               aspect=(N-dt*sampling_rate)*dt/(2*sampling_rate**2))
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Frequency [Hz]')
    plt.show()


if __name__=="__main__":

    N = 1000
    sampling_rate = 100
    dt = 1

    t = np.arange(N)/sampling_rate
    #frequency = 0.08*sampling_rate*(1+np.cos(2*np.pi*0.1*t))  # Sine frequency
    #frequency = t*(sampling_rate**2)/(4*N)  # Linear frequency
    frequency = 0.125*np.power(t, 2)*sampling_rate**3/(1*N**2)  # Quadratic frequency

    signal = np.cos(2*np.pi*frequency*t)

    #plt.plot(t, frequency)
    plt.plot(t, signal)
    plt.xlabel("Time [sec]")
    plt.ylabel("Magnitude")
    plt.show()

    swiping_fixed_chirpiness(signal, sampling_rate, dt)
    swiping_fixed_frequency(signal, sampling_rate, dt)
