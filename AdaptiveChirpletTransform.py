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

    def act_greedy(self, tc=0, dt=None):
        gamma = np.arange(self.length+1)
        T = self.length/self.sampling_rate  # Length of the signal in sec

        fc_range = 0.5*gamma/T  # Center frequency ranges from 0 to sampling_rate/2
        if dt is None:
            dt = self.length/(10*self.sampling_rate)  # Fixed window size: (length of signal in sec)/10
        c_range = np.concatenate((-0.5*gamma/(T*dt), 0.5*gamma/(T*dt)))  # Chirpiness ranges from 0 to sampling_rate/(2*dt)

        def get_Imax(Rnf, tc, fc_range, c_range, dt):
            fc = self.sampling_rate/2
            c = self.sampling_rate/(2*dt)

            sc = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)\
                .chirplet_transform(Rnf)
            delta_sc = sc
            i = 0
            while delta_sc > 1e-3 and i < 100:
                fc = self.best_fitting_frequency(tc, c, signal=Rnf, fc_range=fc_range)
                c = self.best_fitting_chirpiness(tc, fc, dt, signal=Rnf, c_range=c_range)
                new_sc = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)\
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
        tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt)
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet))
            plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            plt.show()
            tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt)
            chirplet_list.append(Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False))

        #print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]#, time.time() - start_time

    def act_exhaustive(self, tc=0, dt=None):
        gamma = np.arange(self.length+1)
        T = self.length/self.sampling_rate  # Length of the signal in sec

        fc_range = 0.5*gamma/T  # Center frequency ranges from 0 to sampling_rate/2
        if dt is None:
            dt = self.length/(10*self.sampling_rate)  # Fixed window size: (length of signal in sec)/10
        c_range = np.concatenate((-0.5*gamma/(T*dt), 0.5*gamma/(T*dt)))  # Chirpiness ranges from 0 to sampling_rate/(2*dt)

        def get_Imax(Rnf, tc, fc_range, c_range, dt):

            def search_best_fc_c(f, tc, fc_range, c_range):
                N, M = len(fc_range), len(c_range)
                t = np.multiply.outer(np.ones(N), np.arange(self.length))
                f_range = np.outer(fc_range, np.ones(self.length))

                scalar_products = np.zeros((M, N))
                for i in range(M):
                    c = c_range[i]

                    chirps = np.exp(2j * np.pi * (c * (t - tc) + f_range) * (t - tc))
                    scalar_products[i] = np.abs(np.sum(np.outer(np.ones(N), f) * np.conj(chirps), axis=1))

                imax = np.argmax(scalar_products)
                fc = fc_range[imax%N]
                c = c_range[imax//N]

                #print(wcmax, cmax, scalar_products[imax//N][imax%N])

                return fc, c

            fc, c = search_best_fc_c(Rnf, tc, fc_range, c_range)
            sc = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)\
                .chirplet_transform(Rnf)

            if sc != sc:
                fc, c = 0, 0

            return tc, fc, c, dt

        #start_time = time.time()

        R = [self.f]
        tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt)
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet))
            plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            plt.show()
            tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt)
            chirplet_list.append(Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False))

        #print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]#, time.time() - start_time

    def act_gradient_descent(self, tc=0, dt=None):
        gamma = np.arange(self.length+1)
        T = self.length/self.sampling_rate  # Length of the signal in sec

        fc_range = 0.5*gamma/T  # Center frequency ranges from 0 to sampling_rate/2
        if dt is None:
            dt = self.length/(10*self.sampling_rate)  # Fixed window size: (length of signal in sec)/10
        c_range = np.concatenate((-0.5*gamma/(T*dt), 0.5*gamma/(T*dt)))  # Chirpiness ranges from 0 to sampling_rate/(2*dt)

        def get_Imax(Rnf, tc_start, fc_start, c_start, dt, alpha=5e-1, beta=0.95, max_iter=5000):

            chirplet = Chirplet(tc=tc_start, fc=fc_start, c=c_start, dt=dt, length=self.length,
                                sampling_rate=self.sampling_rate)
            dtc, dfc, dc = chirplet.derive_transform_wrt_tc(Rnf), chirplet.derive_transform_wrt_fc(
                Rnf), chirplet.derive_transform_wrt_c(Rnf)
            dtc, dfc, dc = np.real(dtc), np.real(dfc), np.real(dc)
            vtc, vfc, vc = 0.1 * alpha * dtc, alpha * dfc, alpha * dc
            if tc_start + vtc < 0 or tc_start + vtc > dt:
                vtc = 0
            if fc_start + vfc < 0 or fc_start + vfc > 0.5 * self.sampling_rate:  # Maximum frequency: sampling_rate/2
                vfc = 0
            if c_start + vc < 0 or c_start + vc > 0.5 * self.sampling_rate / dt:  # Maximum chirpiness: sampling_rate/(2*dt)
                vc = 0
            tc, fc, c = tc_start + vtc, fc_start + vfc, c_start + vc

            nb_iter = 0

            fc_list = [fc_start, fc]
            c_list = [c_start, c]
            tc_list = [tc_start, tc]
            sc_list = [chirplet.chirplet_transform(Rnf)]

            while (np.abs(vfc) >= 1e-4 or np.abs(vc) >= 1e-4) and nb_iter < max_iter:
                chirplet = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate)
                dtc, dfc, dc = chirplet.derive_transform_wrt_tc(Rnf), chirplet.derive_transform_wrt_fc(Rnf), chirplet.derive_transform_wrt_c(Rnf)
                dtc, dfc, dc = np.real(dtc), np.real(dfc), np.real(dc)
                vtc, vfc, vc = beta*vtc + 0.1*alpha*dtc, beta*vfc + alpha*dfc, beta*vc + alpha*dc
                if tc + vtc < 0 or tc + vtc > dt:
                    vtc = 0
                if fc + vfc < 0 or fc + vfc > 0.5*self.sampling_rate:  # Maximum frequency: sampling_rate/2
                    vfc = 0
                if c + vc < 0 or c + vc > 0.5*self.sampling_rate/dt:  # Maximum chirpiness: sampling_rate/(2*dt)
                    vc = 0
                tc, fc, c = tc + vtc, fc + vfc, c + vc
                tc_list.append(tc)
                fc_list.append(fc)
                c_list.append(c)
                sc_list.append(chirplet.chirplet_transform(Rnf))

                nb_iter += 1

            chirplet = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate)
            sc = chirplet.chirplet_transform(Rnf)
            sc_list.append(sc)

            #plt.plot(np.arange(len(tc_list)), [self.sampling_rate*x for x in tc_list], label="Center time")
            #plt.plot(np.arange(len(fc_list)), fc_list, label="Center frequency")
            #plt.plot(np.arange(len(c_list)), c_list, label="Chirpiness")
            #plt.plot(np.arange(len(sc_list)), [self.sampling_rate*x/4 for x in sc_list], label="Chirplet transform")
            #plt.legend()
            #plt.show()

            if sc != sc:
                tc, fc, c = 0, 0, 0

            return tc, fc, c, dt

        #start_time = time.time()

        fc_start = self.sampling_rate/4
        c_start = self.sampling_rate/(4*dt)

        R = [self.f]
        tc, fc, c, dt = get_Imax(R[-1], tc, fc_start, c_start, dt)
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet))
            plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            plt.show()
            tc, fc, c, dt = get_Imax(R[-1], dt / 2, fc_start, c_start, dt)
            chirplet_list.append(Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate))

        #print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]#, time.time() - start_time

    def best_fitting_frequency(self, tc, c, signal=None, fc_range=None):
        if signal is None:
            signal = self.f
        if fc_range is None:
            gamma = np.arange(self.length+1)
            T = self.length/self.sampling_rate  # Length of the signal in sec

            fc_range = 0.5*gamma/T  # Center frequency ranges from 0 to sampling_rate/2

        t = np.outer(np.ones(len(fc_range)), np.arange(self.length)) / self.sampling_rate
        f_range = np.outer(fc_range, np.ones(self.length))

        chirps = np.exp(2j * np.pi * (c * (t - tc) + f_range) * (t - tc))
        scalar_products = np.abs(np.sum(np.outer(np.ones(len(fc_range)), signal) * np.conj(chirps), axis=1))

        fc = fc_range[np.argmax(scalar_products)]

        return fc

    def best_fitting_chirpiness(self, tc, fc, dt, signal=None, c_range=None):
        if signal is None:
            signal = self.f
        if c_range is None:
            gamma = np.arange(self.length+1)
            T = self.length/self.sampling_rate  # Length of the signal in sec

            c_range = np.concatenate(
                (-0.5 * gamma / (T * dt), 0.5 * gamma / (T * dt)))  # Chirpiness ranges from 0 to sampling_rate/(2*dt)

        t = np.outer(np.ones(len(c_range)), np.arange(self.length)) / self.sampling_rate
        c_range_ = np.outer(c_range, np.ones(self.length))

        chirps = np.exp(2j * np.pi * (c_range_ * (t - tc) + fc) * (t - tc))
        scalar_products = np.abs(np.sum(np.outer(np.ones(len(c_range)), signal) * np.conj(chirps), axis=1))

        c = c_range[np.argmax(scalar_products)]

        return c
