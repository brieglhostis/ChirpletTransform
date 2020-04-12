import numpy as np
import matplotlib.pyplot as plt


class Chirplet:

    def __init__(self, tc, fc, c, dt, length, sampling_rate=1, gaussian=True):
        self.tc = tc  # Center time in sec
        self.fc = fc  # Center frequency in Hz
        self.wc = 2*np.pi*fc  # Center pulsation in rad/s
        self.c = c  # Chirpiness in Hz^2
        self.dt = dt  # Window of the chirplet in sec
        self.length = length  # Length of the Chriplet
        self.sampling_rate = sampling_rate  # Sampling frequency of the Chriplet in Hz
        self.I = (tc, fc, c, dt)

        t = np.arange(length)/sampling_rate
        if gaussian:
            window = np.exp(-0.5*np.power((t-tc)/dt, 2))
            magnitude = 1/(np.abs(sampling_rate*dt)*np.pi**0.5)**0.5
        else:
            window = np.zeros(length)
            starting_index = max(0, int(sampling_rate*(tc-dt/2)))
            ending_index = min(length, int(sampling_rate*(tc+dt/2)))
            window[starting_index: ending_index] = 1
            magnitude = 1/(ending_index-starting_index)**0.5

        self.chirplet = magnitude*window*np.exp(2j*np.pi*(c*(t-tc)+fc)*(t-tc))

    def chirplet_transform(self, f):
        return np.abs(np.vdot(f, self.chirplet))

    def wigner_distribution(self, k=4):
        Id = np.ones(k*self.length)
        t = np.outer(Id, np.arange(k*self.length))/(self.sampling_rate*k)
        f = self.sampling_rate*np.outer(np.arange(k*self.length), Id)/(2*self.length*k) # Maximum frequency: sampling_rate/2
        return 2 * np.exp(-np.power((t/k-self.tc)/self.dt, 2)
                          - np.power(2*np.pi*self.dt*(f - self.fc - 2 * self.c * (t-self.tc)), 2))

    def plot_chirplet(self):
        t = np.arange(self.length)/self.sampling_rate
        plt.plot(t, np.real(self.chirplet))
        plt.show()

    def plot_wigner_distribution(self, k=4):
        distribution = self.wigner_distribution(k=k)
        distribution = np.flip(distribution, axis=0) # move minimal frequency at the bottom of the plot
        plt.figure()
        plt.imshow(distribution, cmap="gray", extent=(0, self.length/self.sampling_rate, 0, self.sampling_rate/2),
                   aspect=2*self.length/(self.sampling_rate**2))
        plt.xlabel("Time t [sec]")
        plt.ylabel("Frequency f [Hz]")
        plt.show()

    def get_starting_and_ending_frequencies(self):
        starting_time = max(0, self.tc-self.dt/2)
        ending_time = min(self.length/self.sampling_rate, self.tc+self.dt/2)
        starting_frequency = self.fc + self.c*(starting_time-self.tc)
        ending_frequency = self.fc + self.c*(ending_time-self.tc)
        return starting_frequency, ending_frequency

