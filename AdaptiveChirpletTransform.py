import numpy as np
import matplotlib.pyplot as plt
import time

from Chirplet import Chirplet


class ACT:

    def __init__(self, f, sampling_rate=1, P=1):
        self.f = f
        self.P = P
        self.sampling_rate = sampling_rate
        self.length = len(f)

    def wigner_distribution(self, k=4):
        f = np.concatenate((np.zeros(self.length), self.f, np.zeros(self.length)))
        taus = np.arange(2 * self.length)
        ts = np.arange(k * self.length) / k
        ws = np.arange(k * self.length) / (2 * k * self.length) # Maximum frequency: sampling_rate/2
        idxf = self.length + taus[np.newaxis] / 2 + ts[:, np.newaxis]
        idxfc = self.length - taus[np.newaxis] / 2 + ts[:, np.newaxis]
        unity_roots = np.exp(-1j * 2 * np.pi * np.outer(taus, ws))
        return np.dot(f[idxf.astype(np.int)] * np.conj(f[idxfc.astype(np.int)]), unity_roots).T

    def plot_wigner_distribution(self, k=4):
        distribution = np.abs(self.wigner_distribution(k=k))
        distribution = np.flip(distribution, axis=0) # move minimal frequency at the bottom of the plot
        plt.figure()
        plt.imshow(distribution, cmap="gray", extent=(0, self.length/self.sampling_rate, 0, self.sampling_rate/2),
                   aspect=2*self.length/(self.sampling_rate**2))
        plt.xlabel("Time t [sec]")
        plt.ylabel("Frequency f [Hz]")
        plt.show()

    def act_greedy(self, tc=0, dt=None, gaussian_chirplet=True):
        gamma = np.arange(self.length+1)
        T = self.length/self.sampling_rate  # Length of the signal in sec

        fc_range = 0.5*gamma/T  # Center frequency ranges from 0 to sampling_rate/2
        if dt is None:
            dt = self.length/(10*self.sampling_rate)  # Fixed window size: (length of signal in sec)/10
        c_range = np.concatenate((-0.5*gamma/(T*dt), 0.5*gamma/(T*dt)))  # Chirpiness ranges from 0 to sampling_rate/(2*dt)

        def get_Imax(Rnf, tc, fc_range, c_range, dt, gaussian_chirplet=True):
            fc = self.sampling_rate/4
            c = self.sampling_rate/(4*dt)

            # Find the best center frequency in the dictionary with chirpiness c=0 and maximum window size

            def search_best_frequency(f, tc, fc_range, c, dt, gaussian_chirplet=True):
                t = np.outer(np.ones(len(fc_range)), np.arange(self.length))/self.sampling_rate
                f_range = np.outer(fc_range, np.ones(self.length))

                if gaussian_chirplet:
                    window = np.exp(-np.power((t-tc)/dt, 2))
                    magnitude = 1/np.power(np.abs(dt)*np.power(np.pi, 0.5), 0.5)
                else:
                    window = np.zeros(self.length)
                    starting_index = max(0, int(self.sampling_rate*(tc-dt/2)))
                    ending_index = min(self.length, int(self.sampling_rate*(tc+dt/2)))
                    window[starting_index: ending_index] = 1
                    magnitude = 1/(ending_index-starting_index)

                chirplets = magnitude*window*np.exp(2j*np.pi*(c*(t-tc)+f_range)*(t-tc))
                scalar_products = np.abs(np.sum(np.outer(np.ones(len(fc_range)), f) * np.conj(chirplets), axis=1))

                fc = fc_range[np.argmax(scalar_products)]

                return fc

            def search_best_chirpiness(f, tc, fc, c_range, dt, gaussian_chirplet=True):
                t = np.outer(np.ones(len(c_range)), np.arange(self.length))/self.sampling_rate
                c_range_ = np.outer(c_range, np.ones(self.length))

                if gaussian_chirplet:
                    window = np.exp(-np.power((t-tc)/dt, 2))
                    magnitude = 1/np.power(np.abs(dt)*np.power(np.pi, 0.5), 0.5)
                else:
                    window = np.zeros(self.length)
                    starting_index = max(0, int(self.sampling_rate*(tc-dt/2)))
                    ending_index = min(self.length, int(self.sampling_rate*(tc+dt/2)))
                    window[starting_index: ending_index] = 1
                    magnitude = 1/(ending_index-starting_index)

                chirplets = magnitude*window*np.exp(2j*np.pi*(c_range_*(t-tc)+fc)*(t-tc))
                scalar_products = np.abs(np.sum(np.outer(np.ones(len(c_range)), f) * np.conj(chirplets), axis=1))

                c = c_range[np.argmax(scalar_products)]

                return c

            sc = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=gaussian_chirplet)\
                .chirplet_transform(Rnf)
            delta_sc = sc
            i = 0
            while delta_sc > 1e-3 and i < 100:
                fc = search_best_frequency(Rnf, tc, fc_range, c, dt, gaussian_chirplet=gaussian_chirplet)
                c = search_best_chirpiness(Rnf, tc, fc, c_range, dt, gaussian_chirplet=gaussian_chirplet)
                new_sc = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=gaussian_chirplet)\
                    .chirplet_transform(Rnf)
                delta_sc = np.abs(sc - new_sc)
                sc = new_sc
                i += 1
                #print(i, fc, c, sc)

            if sc != sc:
                fc, c = 0, 0

            return tc, fc, c, dt

        #start_time = time.time()

        R = [self.f]
        tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt, gaussian_chirplet=gaussian_chirplet)
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=gaussian_chirplet)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet))
            plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            plt.show()
            tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt, gaussian_chirplet=gaussian_chirplet)
            chirplet_list.append(Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=gaussian_chirplet))

        #print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]#, time.time() - start_time

    def act_exhaustive(self, tc=0, dt=None, gaussian_chirplet=True):
        gamma = np.arange(self.length+1)
        T = self.length/self.sampling_rate  # Length of the signal in sec

        fc_range = 0.5*gamma/T  # Center frequency ranges from 0 to sampling_rate/2
        if dt is None:
            dt = self.length/(10*self.sampling_rate)  # Fixed window size: (length of signal in sec)/10
        c_range = np.concatenate((-0.5*gamma/(T*dt), 0.5*gamma/(T*dt)))  # Chirpiness ranges from 0 to sampling_rate/(2*dt)

        def get_Imax(Rnf, tc, fc_range, c_range, dt, gaussian_chirplet=True):

            def search_best_fc_c(f, tc, fc_range, c_range, dt, gaussian_chirplet=True):
                N, M = len(fc_range), len(c_range)
                t = np.multiply.outer(np.ones(N), np.arange(self.length))
                f_range = np.outer(fc_range, np.ones(self.length))

                if gaussian_chirplet:
                    window = np.exp(-np.power((t-tc)/dt, 2))
                    magnitude = 1/np.power(np.abs(dt)*np.power(np.pi, 0.5), 0.5)
                else:
                    window = np.zeros(self.length)
                    starting_index = max(0, int(self.sampling_rate*(tc-dt/2)))
                    ending_index = min(self.length, int(self.sampling_rate*(tc+dt/2)))
                    window[starting_index: ending_index] = 1
                    magnitude = 1/(ending_index-starting_index)

                scalar_products = np.zeros((M, N))
                for i in range(M):
                    c = c_range[i]

                    chirplets = magnitude * window * np.exp(2j * np.pi * (c * (t - tc) + f_range) * (t - tc))
                    scalar_products[i] = np.abs(np.sum(np.outer(np.ones(N), f) * np.conj(chirplets), axis=1))

                imax = np.argmax(scalar_products)
                fc = fc_range[imax%N]
                c = c_range[imax//N]

                #print(wcmax, cmax, scalar_products[imax//N][imax%N])

                return fc, c

            fc, c = search_best_fc_c(Rnf, tc, fc_range, c_range, dt, gaussian_chirplet=gaussian_chirplet)
            sc = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=gaussian_chirplet)\
                .chirplet_transform(Rnf)

            if sc != sc:
                fc, c = 0, 0

            return tc, fc, c, dt

        #start_time = time.time()

        R = [self.f]
        tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt, gaussian_chirplet=gaussian_chirplet)
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=gaussian_chirplet)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet))
            plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            plt.show()
            tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt, gaussian_chirplet=gaussian_chirplet)
            chirplet_list.append(Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=gaussian_chirplet))

        #print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]#, time.time() - start_time
