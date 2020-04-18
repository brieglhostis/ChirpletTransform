import numpy as np
import matplotlib.pyplot as plt
import time

from ChirpletTransform.Chirplet import Chirplet


class ACT:

    def __init__(self, f, sampling_rate=1.0, P=1):
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

    def plot_wigner_distribution(self, k=4, title=None):
        distribution = np.abs(self.wigner_distribution(k=k))
        distribution = np.flip(distribution, axis=0) # move minimal frequency at the bottom of the plot
        plt.figure()
        plt.imshow(distribution, cmap="gray", extent=(0, self.length/self.sampling_rate, 0, self.sampling_rate/2),
                   aspect=2*self.length/(self.sampling_rate**2))
        plt.xlabel("Time [sec]")
        plt.ylabel("Frequency [Hz]")
        if isinstance(title, str):
            plt.title(title)
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

        start_time = time.time()

        R = [self.f]
        tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt)
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet))
            #plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            #plt.show()
            tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt)
            chirplet_list.append(Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False))

        print("time elapsed: {:.2f}s".format(time.time() - start_time))

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

        start_time = time.time()

        R = [self.f]
        tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt)
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet))
            #plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            #plt.show()
            tc, fc, c, dt = get_Imax(R[-1], tc, fc_range, c_range, dt)
            chirplet_list.append(Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False))

        print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]#, time.time() - start_time

    def act_gradient_descent(self, tc=0, dt=None, alpha=5e-1, beta=0.95, max_iter=5000, fixed_tc=True, print_time=True):
        gamma = np.arange(self.length+1)
        T = self.length/self.sampling_rate  # Length of the signal in sec

        fc_range = 0.5*gamma/T  # Center frequency ranges from 0 to sampling_rate/2
        if dt is None:
            dt = self.length/(10*self.sampling_rate)  # Fixed window size: (length of signal in sec)/10
        c_range = np.concatenate((-0.5*gamma/(T*dt), 0.5*gamma/(T*dt)))  # Chirpiness ranges from 0 to sampling_rate/(2*dt)

        def get_Imax(Rnf, tc_start, fc_start, c_start, dt, alpha=5e-1, beta=0.95, max_iter=5000, fixed_tc=True):

            alpha_tc = 0.2*alpha*dt*self.sampling_rate/len(Rnf)

            sc = Chirplet(tc=tc_start, fc=fc_start, c=c_start, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)\
                .chirplet_transform(Rnf)
            delta_sc = sc
            i = 0
            while delta_sc > 1e-3 and i < 100:
                fc_start = self.best_fitting_frequency(tc_start, c_start, signal=Rnf, fc_range=fc_range)
                c_start = self.best_fitting_chirpiness(tc_start, fc_start, dt, signal=Rnf, c_range=c_range)
                new_sc = Chirplet(tc=tc_start, fc=fc_start, c=c_start, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False)\
                    .chirplet_transform(Rnf)
                delta_sc = np.abs(sc - new_sc)
                sc = new_sc
                i += 1

            chirplet = Chirplet(tc=tc_start, fc=fc_start, c=c_start, dt=dt, length=self.length,
                                sampling_rate=self.sampling_rate)
            dtc, dfc, dc = chirplet.derive_transform_wrt_tc(Rnf), chirplet.derive_transform_wrt_fc(
                Rnf), chirplet.derive_transform_wrt_c(Rnf)
            dtc, dfc, dc = np.real(dtc), np.real(dfc), np.real(dc)
            vtc, vfc, vc = alpha_tc * dtc, alpha * dfc, alpha * dc
            if tc_start + vtc < 0 or tc_start + vtc > dt:
                vtc = 0
            if fc_start + vfc < 0 or fc_start + vfc > 0.5 * self.sampling_rate:  # Maximum frequency: sampling_rate/2
                vfc = 0
            if c_start + vc < 0 or c_start + vc > 0.5 * self.sampling_rate / dt:  # Maximum chirpiness: sampling_rate/(2*dt)
                vc = 0
            fc, c = fc_start + vfc, c_start + vc
            if not fixed_tc:
                tc = tc_start + vtc
            else:
                tc = tc_start

            nb_iter = 0

            fc_list = [fc_start, fc]
            c_list = [c_start, c]
            tc_list = [tc_start, tc]
            sc_list = [chirplet.chirplet_transform(Rnf)]

            while (np.abs(vfc) >= 1e-4 or np.abs(vc) >= 1e-4) and nb_iter < max_iter:
                chirplet = Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate)
                dtc, dfc, dc = chirplet.derive_transform_wrt_tc(Rnf), chirplet.derive_transform_wrt_fc(Rnf), chirplet.derive_transform_wrt_c(Rnf)
                dtc, dfc, dc = np.real(dtc), np.real(dfc), np.real(dc)
                vtc, vfc, vc = beta*vtc + alpha_tc*dtc, beta*vfc + alpha*dfc, beta*vc + alpha*dc
                if tc + vtc < 0 or tc + vtc > dt:
                    vtc = 0
                if fc + vfc < 0 or fc + vfc > 0.5*self.sampling_rate:  # Maximum frequency: sampling_rate/2
                    vfc = 0
                if c + vc < 0 or c + vc > 0.5*self.sampling_rate/dt:  # Maximum chirpiness: sampling_rate/(2*dt)
                    vc = 0
                fc, c = fc + vfc, c + vc
                if not fixed_tc:
                    tc = tc + vtc
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

        start_time = time.time()

        fc_start = self.sampling_rate/4
        c_start = self.sampling_rate/(4*dt)

        R = [self.f]
        tc, fc, c, dt = get_Imax(R[-1], tc, fc_start, c_start, dt, alpha=alpha, beta=beta, max_iter=max_iter,
                                 fixed_tc=fixed_tc)
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f),
                                         np.real(chirplet_list[-1].chirplet)))
            #plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            #plt.show()
            tc, fc, c, dt = get_Imax(R[-1], tc, fc_start, c_start, dt, alpha=alpha, beta=beta, max_iter=max_iter,
                                     fixed_tc=fixed_tc)
            chirplet_list.append(Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate))

        if print_time:
            print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]

    def act_steve_mann(self, t0=0, dt=None, max_iter=500, starting_frequencies=None, ending_frequencies=None):

        if dt is None:
            dt = self.length/(10*self.sampling_rate)  # Fixed window size: (length of signal in sec)/10

        def chirplet_transform(f, t0, starting_frequency, ending_frequency, dt):
            c = (ending_frequency - starting_frequency)/dt
            chirplet = Chirplet(tc=t0+dt/2, fc=starting_frequency+c*dt/2, c=c, dt=dt, length=len(f),
                                sampling_rate=self.sampling_rate, gaussian=False)
            return chirplet.chirplet_transform(f)

        def maximize_similarity(f, t0, starting_frequencies, ending_frequencies, dt, max_iter=500):

            f = np.array(f)
            f = f - np.mean(f)

            for iter in range(max_iter):

                # Find the worst parameters to eliminate them:
                index = np.argmin([chirplet_transform(f, t0, starting_frequencies[0], ending_frequencies[1], dt),
                                   chirplet_transform(f, t0, starting_frequencies[1], ending_frequencies[1], dt),
                                   chirplet_transform(f, t0, starting_frequencies[2], ending_frequencies[1], dt)])

                if index == 0:
                    starting_frequencies = [starting_frequencies[1], starting_frequencies[2],
                                            2 * starting_frequencies[2] - starting_frequencies[1]]
                else:
                    starting_frequencies = [max(2 * starting_frequencies[0] - starting_frequencies[1], 0),
                                            starting_frequencies[0], starting_frequencies[1]]

                index = np.argmin([chirplet_transform(f, t0, starting_frequencies[1], ending_frequencies[0], dt),
                                   chirplet_transform(f, t0, starting_frequencies[1], ending_frequencies[1], dt),
                                   chirplet_transform(f, t0, starting_frequencies[1], ending_frequencies[2], dt)])

                if index == 0:
                    ending_frequencies = [ending_frequencies[1], ending_frequencies[2],
                                          2 * ending_frequencies[2] - ending_frequencies[1]]
                else:
                    ending_frequencies = [max(2 * ending_frequencies[0] - ending_frequencies[1], 0),
                                          ending_frequencies[0], ending_frequencies[1]]

                # Compress inwards
                if iter % 2 == 0:
                    starting_frequencies = [0.5 * (starting_frequencies[0] + starting_frequencies[1]),
                                            starting_frequencies[1],
                                            0.5 * (starting_frequencies[1] + starting_frequencies[2])]
                    ending_frequencies = [0.5 * (ending_frequencies[0] + ending_frequencies[1]),
                                          ending_frequencies[1],
                                          0.5 * (ending_frequencies[1] + ending_frequencies[2])]

            return starting_frequencies[1], ending_frequencies[1]

        start_time = time.time()

        if starting_frequencies is None:
            starting_frequencies = [-0.25*self.sampling_rate, 0, 0.25*self.sampling_rate]
        if ending_frequencies is None:
            ending_frequencies = [-0.25*self.sampling_rate, 0, 0.25*self.sampling_rate]

        R = [self.f]
        starting_frequency, ending_frequency = maximize_similarity(R[-1], t0, starting_frequencies, ending_frequencies,
                                                                   dt, max_iter=max_iter)
        f0 = starting_frequency
        c = (ending_frequency-starting_frequency)/dt
        chirplet_list = [Chirplet(tc=t0+dt/2, fc=f0+c*dt/2, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate,
                                  gaussian=False)]

        for n in range(1, self.P):
            R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet))
            #plt.plot(np.arange(len(R[-1])) / self.sampling_rate, np.real(R[-1]))
            #plt.show()
            starting_frequency, ending_frequency = maximize_similarity(R[-1], t0, starting_frequencies,
                                                                       ending_frequencies, dt, max_iter=max_iter)
            f0 = starting_frequency
            c = (ending_frequency - starting_frequency)/dt
            chirplet_list.append(
                Chirplet(tc=t0+dt/2, fc=f0+c*dt/2, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate, gaussian=False))

        print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]  # , time.time() - start_time

    def act_mp_lem(self, tc=0, dt=None, delta_1=1e-2, delta_2=1):

        if dt is None:
            dt = self.length/(10*self.sampling_rate)  # Fixed window size: (length of signal in sec)/10

        start_time = time.time()

        R = [self.f]
        act = ACT(R[-1], sampling_rate=self.sampling_rate, P=1)
        tc, fc, c, dt = act.act_gradient_descent(tc=tc, dt=dt, fixed_tc=False, print_time=False)[0]
        chirplet_list = [Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate)]
        R.append(R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f),
                                     np.real(chirplet_list[-1].chirplet)))
        p = 1
        cc = np.abs(chirplet_list[-1].chirplet_transform(self.f))**2/np.abs(np.vdot(R[-1], R[-1]))
        print(cc)
        while p < self.P and cc < delta_2:
            p += 1
            # plt.plot(np.arange(len(R[-1]))/self.sampling_rate, np.real(R[-1]))
            # plt.show()
            act = ACT(R[-1], sampling_rate=self.sampling_rate, P=1)
            tc, fc, c, dt = act.act_gradient_descent(tc=tc, dt=dt, fixed_tc=False, print_time=False)[0]
            chirplet_list.append(
                Chirplet(tc=tc, fc=fc, c=c, dt=dt, length=self.length, sampling_rate=self.sampling_rate))

            error = R[-1] - np.multiply(chirplet_list[-1].chirplet_transform(self.f), chirplet_list[-1].chirplet)
            norm_error = np.abs(np.vdot(error, error))**0.5
            nb_iter = 0

            while norm_error > delta_1 and nb_iter < 20:
                new_chirplet_list = []
                new_error = self.f
                for k in range(p):
                    a_k = chirplet_list[k].chirplet_transform(self.f)
                    g_k = chirplet_list[k].chirplet
                    y_k = a_k*g_k + error/p
                    act_k = ACT(y_k, sampling_rate=self.sampling_rate, P=1)
                    I_k = act_k.act_gradient_descent(tc=chirplet_list[k].tc, dt=dt, fixed_tc=False, print_time=False)[0]
                    new_chirplet_list.append(Chirplet(tc=I_k[0], fc=I_k[1], c=I_k[2], dt=I_k[3], length=self.length,
                                                      sampling_rate=self.sampling_rate))
                    new_error -= np.multiply(new_chirplet_list[-1].chirplet_transform(self.f),
                                             np.real(new_chirplet_list[-1].chirplet))
                    new_norm_error = np.abs(np.vdot(new_error, new_error))**0.5
                if new_norm_error > norm_error:
                    print("Break at iteration number", nb_iter+1)
                    break
                else:
                    chirplet_list = new_chirplet_list[:]
                    error = np.copy(new_error)
                    norm_error = np.abs(np.vdot(error, error))**0.5
                nb_iter += 1
                print(p, nb_iter, "error:", norm_error)
            R.append(error)

            cc = np.abs(chirplet_list[-1].chirplet_transform(self.f)) ** 2 / np.abs(np.vdot(R[-1], R[-1]))
            print(p, "cc:", cc)

        print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return [chirplet.I for chirplet in chirplet_list]

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
