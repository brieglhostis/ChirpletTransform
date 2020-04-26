import numpy as np
import matplotlib.pyplot as plt
import time

def WVD(f, k=4):
    '''
    Wigner-Viller Distribution
    Used to transform a signal into time-frequency domain
    '''
    N = len(f)
    f = np.array([0 for i in range(N)] + f + [0 for i in range(N)])
    taus = np.arange(2*N)
    ts = np.arange(k*N)/k
    ws = np.arange(k*N)/(k*N)
    idxf = N + taus[np.newaxis]/2 + ts[:,np.newaxis]
    idxfc = N - taus[np.newaxis]/2 + ts[:,np.newaxis]
    unity_roots = np.exp(-1j*2*np.pi*np.outer(taus,ws))
    return np.dot(f[idxf.astype(np.int)]*np.conj(f[idxfc.astype(np.int)]),unity_roots).T

class ACT(object):
    '''
    Base Adaptive Chirplet Transform class
    '''
    @classmethod
    def gaussian_chirplet(cls, t, I):
        tc, wc, c, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*(c*(t-tc)+wc)*(t-tc)
        )

    @classmethod
    def gaussian_chirlet_transform(cls, f, I):
        t = np.arange(len(f))
        return np.abs(np.dot(f, np.conj(cls.gaussian_chirplet(t, I))))

    @classmethod
    def WVgI(cls, I, N, k=4):
        tc, wc, c, dt = I
        t = np.arange(k*N)
        w = np.arange(k*N)[:,None]
        return 2*np.exp(-np.power((t/k-tc)/dt,2)-np.power(dt*((2*np.pi*w/N)/k-wc-2*c*(t/k-tc)),2))

    @classmethod
    def kernel(cls, dt, c, t):
        return np.exp(-0.5*np.power(t/dt, 2) + 1j*c*t**2)/np.power(np.power(np.pi, 0.5)*dt, 0.5)

    @classmethod
    def norm_l2(cls, fct):
        return np.abs(np.dot(fct,fct)**2)

    @classmethod
    def first_derivate_g(cls, g, I):
        tc, wc, c, dt = I
        t = np.arange(len(g))
        dtc = ((t-tc)/(dt**2) - 1.j*c*(t-tc) -1.j*wc)*g
        dwc = 1.j*(t-tc)*g

        return dtc, dwc

    @classmethod
    def f_prime_f_second(cls, Rnf, g, I):
        dtc, dwc = cls.first_derivate_g(g, I)

        sc = np.dot(Rnf, np.conj(g))

        scdtc = np.dot(Rnf, np.conj(dtc))
        dfdtc = (scdtc*np.conj(sc) + sc*np.conj(scdtc))/(2*np.abs(sc))

        scdwc = np.dot(Rnf, np.conj(dwc))
        dfdwc = scdwc*np.conj(sc) + sc*np.conj(scdwc)/(2*np.abs(sc))


        return dfdtc, dfdwc

    @classmethod
    def get_Imax(cls, Rnf, c, dt, max_iter = 10000, alpha = 1):
        N = len(Rnf)
        _t = np.arange(N)
        cmax = 0
        dtmax = 1
        scmax = 0
        wcmax = 0
        tc_ = np.argmax(np.abs(Rnf))
        for wc_ in 2*np.pi*np.arange(10)/10:
            for k in range(len(dt)):
                dt_ = dt[k]
                for c_ in c[k]:
                    ker = cls.kernel(dt_, c_, _t-tc_)*np.exp(-1j*_t*wc_)
                    sc = np.abs(np.dot(Rnf,ker))
                    if sc>scmax:
                        cmax = c_
                        dtmax = dt_
                        scmax = sc
                        wcmax = wc_
        c_, dt_, wc_ = cmax, dtmax, wcmax
        I = tc_, wc_, c_, dt_
        g = cls.gaussian_chirplet(_t, I)
        dtc, dwc = cls.f_prime_f_second(Rnf, g, I)
        dtc, dwc = np.real(dtc), np.real(dwc)
        vtc, vwc = 0.1*alpha * dtc, alpha * dwc
        nb_iter = 0
        tcmax = tc_
        wcmax = wc_
        while (np.abs(vtc) >= 1e-3 or N*np.abs(vwc)/(2*np.pi) >= 1e-3 or nb_iter == 0) and nb_iter<max_iter:
            dtc, dwc = cls.f_prime_f_second(Rnf, g, I)
            dtc, dwc =  np.real(dtc), np.real(dwc)
            vtc, vwc = 0.9*vtc + 0.1*alpha * dtc, 0.9*vwc + alpha * dwc
            if tc_ + vtc < 0 or tc_ + vtc > N:
                vtc = 0
            if wc_ + 2*np.pi*N*vwc/(2*np.pi)/N < 0 or wc_ + 2*np.pi*N*vwc/(2*np.pi)/N > 2*np.pi:
                vwc = 0
            tc_, wc_ = tc_ + vtc, wc_ + 2*np.pi*N*vwc/(2*np.pi)/N
            I = tc_, wc_, c_, dt_
            g = cls.gaussian_chirplet(_t, I)
            nb_iter += 1
            sc = np.abs(np.dot(Rnf,np.conj(g)))
            if sc>scmax:
                tcmax = tc_
                wcmax = wc_
                scmax = sc
        print("GD converged after {} iterations.".format(nb_iter))
        tc_ = tcmax
        wc_ = wcmax
        sc = scmax
        for k in range(len(dt)):
            dt_ = dt[k]
            for c_ in c[k]:
                ker = cls.kernel(dt_, c_, _t-tc_)*np.exp(-1j*_t*wc_)
                sc = np.abs(np.dot(Rnf,ker))
                if sc > scmax:
                    cmax = c_
                    dtmax = dt_
                    scmax = sc
        Imax = tc_, wc_, cmax, dtmax
        print("scalar product :",scmax)
        return Imax

class ACT_MP(ACT):
    '''
    ACT with Matching Pursuit
    Example usage: `ACT_MP.fit(data,P=10)`
    '''
    @classmethod
    def fit(cls, f, P, max_iter_GD = 1000):
        N = len(f)
        _t = np.arange(N)
        i0 = 1
        a = 2
        gamma = np.arange(N+1)
        T = N
        F = 2*np.pi
        D = int(0.5*np.log(N)/np.log(a))
        k = np.arange(D-i0+1)
        m_k = 4*np.power(a, 2*k)-1
        m = [np.arange(mk) for mk in m_k]

        tc = gamma
        wc = (2*np.pi/N)*gamma
        c = [[F*m[k_][i]/(T*np.power(a, 2*k_)) for i in range(m_k[k_])] for k_ in k]
        dt = np.power(a, 2*k)

        start_time = time.time()

        R = [f]
        I = cls.get_Imax(R[-1], c, dt, max_iter=max_iter_GD)
        I_list = [I]
        print("I :", I)
        print("step 1")

        sum_ = np.multiply(np.power(np.abs(cls.gaussian_chirlet_transform(f, I)), 2), cls.WVgI(I,len(f)))

        for n in range(1, P):
            R.append(R[-1]-np.multiply(cls.gaussian_chirlet_transform(f, I), cls.gaussian_chirplet(_t, I)))
            I = cls.get_Imax(R[-1], c, dt, max_iter=max_iter_GD)
            I_list.append(I)
            plt.figure(figsize=(20,2))
            plt.plot(np.linspace(0, len(f), len(f)), np.real(R[-1]))
            plt.show()
           # print("a = ",cls.gaussian_chirlet_transform(f,I))
           # plt.plot(np.multiply(cls.gaussian_chirlet_transform(f, I),
           #                      cls.gaussian_chirplet(_t, I)))
           # plt.show()
           # plt.plot(cls.gaussian_chirplet(_t, I))
           # plt.show()
            print("I :",I)
            print("step " + str(n+1))

            sum_ = np.add(sum_, np.multiply(np.power(np.abs(cls.gaussian_chirlet_transform(f, I)), 2), cls.WVgI(I,len(f))))

        print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return sum_, I_list

class ACT_MP_LEM(ACT):
    '''
    ACT with Matching Pursuit and Logon Expectation Maximization
    Example usage: `ACT_MP_LEM.fit(data,P=10)`
    '''
    @classmethod
    def fit(cls, f, P, threshold1 = 1e-3, threshold2 = 1e-12, iter_max1 = 200, iter_max2 = 10):
        N = len(f)
        _t = np.arange(N)
        i0 = 1
        a = 2
        gamma = np.arange(N+1)
        T = N
        F = 2*np.pi
        D = int(0.5*np.log(N)/np.log(a))
        k = np.arange(D-i0+1)
        m_k = np.array([4*np.power(a, 2*k_)-1 for k_ in k])
        m = [np.arange(mk) for mk in m_k]

        tc = gamma
        wc = (2*np.pi/N)*gamma
        c = [[F*m[k_][i]/(T*np.power(a, 2*k_)) for i in range(m_k[k_])] for k_ in k]
        dt = [np.power(a, 2*k_) for k_ in k]
        
        start_time = time.time()

        R = [f]
        I = cls.get_Imax(R[-1], c, dt, max_iter=iter_max1)
        I_list = [I]
        p = 1
        cc = np.power(np.abs(cls.gaussian_chirlet_transform(f, I))/cls.norm_l2(R[-1]),2)
        print("cc :",cc)
        print("I :",I)
        print("step 1")
        while cc > threshold2 and p < P:
            R.append(np.add(R[-1],-np.multiply(cls.gaussian_chirlet_transform(f, I_list[-1]),
                                               cls.gaussian_chirplet(_t, I_list[-1]))))
            I = cls.get_Imax(R[-1], c, dt, max_iter=iter_max1)
            I_list.append(I)
            e = R[-1]
            nb_iter = 0
            I_min = I_list
            e_min = e
            norm_min = cls.norm_l2(e)
            print("norm(e) before maximization :", cls.norm_l2(e))
            while cls.norm_l2(e) > threshold1 and nb_iter < iter_max2:
                new_I = []
                new_e = f
                for k in range(p+1):
                    a_k = cls.gaussian_chirlet_transform(f, I_list[k])
                    g_k = cls.gaussian_chirplet(_t, I_list[k])
                    y_k = np.add(np.multiply(a_k, g_k), np.multiply(1/p, e))
                    I_k = cls.get_Imax(y_k, c, dt, max_iter=iter_max1)
                    new_I.append(I_k)
                    new_e = np.add(new_e, np.multiply(-cls.gaussian_chirlet_transform(f, I_k), cls.gaussian_chirplet(_t, I_k)))
                e = new_e
                nb_iter += 1
                if cls.norm_l2(e) < norm_min:
                    I_min = new_I
                    e_min = e
                    norm_min = cls.norm_l2(e)
            R[-1] = e_min
            I_list = I_min
            cc = np.power(np.abs(cls.gaussian_chirlet_transform(f, I_list[-1])) / cls.norm_l2(R[-1]), 2)
            p+=1
            #plt.plot(np.linspace(0, len(f), len(f)), [np.real(R[-1][t]) for t in range(len(f))])
            #plt.show()
            print("norm(e) after maximization :", cls.norm_l2(R[-1]))
            print("cc :",cc)
            print("I :", I_list[-1])
            print("step " + str(p))

        sum_ = np.multiply(np.power(np.abs(cls.gaussian_chirlet_transform(f, I_list[0])), 2), cls.WVgI(I_list[0],len(f)))
        for k in range(1, p):
            sum_ = np.add(sum_, np.multiply(np.power(np.abs(cls.gaussian_chirlet_transform(f, I_list[k])), 2), 
                                            cls.WVgI(I_list[k],len(f))))

        print("time elapsed: {:.2f}s".format(time.time() - start_time))

        return sum_, I_list

