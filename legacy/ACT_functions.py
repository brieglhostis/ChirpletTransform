import numpy as np
import matplotlib.pyplot as plt
import time

def WVD(f, k=4):
    N = len(f)
    f = np.array([0 for i in range(N)] + f + [0 for i in range(N)])
    taus = np.arange(2*N)
    ts = np.arange(k*N)/k
    ws = np.arange(k*N)/(k*N)
    idxf = N + taus[np.newaxis]/2 + ts[:,np.newaxis]
    idxfc = N - taus[np.newaxis]/2 + ts[:,np.newaxis]
    unity_roots = np.exp(-1j*2*np.pi*np.outer(taus,ws))
    return np.dot(f[idxf.astype(np.int)]*np.conj(f[idxfc.astype(np.int)]),unity_roots).T

def ACT(f, P, max_iter_GD =200):
    # This function is the "regular" algortihm, it first searches for wc then refines wc and tc using gradient descent and finally searches for the chirpiness c
    N = len(f)
    i0 = 1
    a = 2
    gamma = np.arange(N+1)
    T = N
    F = 2*np.pi
    D = int(np.log(N)/np.log(a))
    k = np.arange(int(0.5*D)-i0+1)
    m_k = np.array([4*np.power(a, 2*k_)-1 for k_ in k])
    m = [np.arange(mk) for mk in m_k]

    wc = (2*np.pi/N)*gamma
    c = [F*m[-1][int(i/4)]*(i%4 + 1)/(T*40*np.power(a, 2*k[-1])) for i in range(4*m_k[-1])]
    dt = [np.power(a, k_) for k_ in range(D)]

    def gaussian_chirplet(t, I):
        tc, wc, c, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*(c*(t-tc)+wc)*(t-tc)
        )

    def gaussian_chirlet_transform(f, I):
        return np.abs(np.vdot(f, gaussian_chirplet(np.arange(len(f)), I)))

    def get_Imax(Rnf, c, max_iter = 5000, alpha = 3e-3):

        def first_derivate_g(g, I):
            tc, wc, c, dt = I
            t = np.arange(len(g))
            dtc = ((t-tc)/(dt*dt) - 2.j*c*(t-tc) -1.j*wc)*g
            dwc = 1.j*(t-tc)*g

            return dtc, dwc

        def f_prime(Rnf, g, I):
            dtc, dwc = first_derivate_g(g, I)

            sc = np.vdot(Rnf, g)

            scdtc = np.vdot(Rnf, dtc)
            dfdtc = scdtc*np.conj(sc) + sc*np.conj(scdtc)

            scdwc = np.vdot(Rnf, dwc)
            dfdwc = scdwc*np.conj(sc) + sc*np.conj(scdwc)

            return 0.5*dfdtc/np.abs(sc), 0.5*dfdwc/np.abs(sc)

        N = len(Rnf)
        I = np.argmax(Rnf), np.pi, 0, a**D
        tc_, wc_, c_, dt_, = I

        # Find the best center frequency in the dictionnary with chirpiness c=0 and maximum window size

        wcmax = I[1]
        scmax = gaussian_chirlet_transform(Rnf, I)

        for wc_ in wc:
            I = tc_, wc_, c_, dt_
            sc = gaussian_chirlet_transform(Rnf, I)
            if sc > scmax:
                scmax = sc
                wcmax = wc_

        wc_ = wcmax
        I = tc_, wc_, c_, dt_

        # Refine tc and wc with chirpiness =0 and maximum window size

        g = [gaussian_chirplet(t, I) for t in range(len(f))]
        dtc, dwc = f_prime(Rnf, g, I)
        dtc, dwc = np.real(dtc), np.real(dwc)
        vtc, vwc = alpha * dtc, alpha * dwc
        nb_iter = 0

        while (np.abs(vtc) >= 1e-4 or N*np.abs(vwc)/(2*np.pi) >= 1e-4 or nb_iter%1000 in [0,1]) and nb_iter<max_iter:
            dtc, dwc = f_prime(Rnf, g, I)
            dtc, dwc =  np.real(dtc), np.real(dwc)
            vtc, vwc = 0.9*vtc + alpha * dtc, 0.9*vwc + alpha * dwc
            if tc_ + vtc < 0 or tc_ + vtc > N:
                vtc = 0
            if wc_ + vwc< 0 or wc_ + vwc > 2*np.pi:
                vwc = 0
            tc_, wc_ = tc_ + vtc, wc_ + vwc

            I = tc_, wc_, c_, dt_
            g = [gaussian_chirplet(t, I) for t in range(len(f))]
            nb_iter += 1

        # Find the best chirpiness in the dictinary with maximum window size

        cmax = I[2]
        scmax = gaussian_chirlet_transform(Rnf, I)

        for c_ in c:
            I = tc_, wc_, c_, dt_
            sc = gaussian_chirlet_transform(Rnf, I)
            if sc > scmax:
                scmax = sc
                cmax = c_

        c_ = cmax
        I = tc_, wc_, c_, dt_

        scmax = gaussian_chirlet_transform(Rnf, I)
        #dtmax = dt_

        #for dt_ in dt:
        #    I = tc_, wc_, c_, dt_
        #    sc = gaussian_chirlet_transform(Rnf, I)
        #    if sc > scmax:
        #        dtmax = dt_
        #        scmax = sc

        #I = tc_, wc_, c_, dtmax

        if scmax != scmax:
            I = 0, 0, 0, dt_

        return I

    start_time = time.time()

    R = [f]
    I = get_Imax(R[-1], c, max_iter=max_iter_GD)
    I_list = [I]

    for n in range(1, P):
        R.append(R[-1] - np.multiply(gaussian_chirlet_transform(f, I), gaussian_chirplet(np.arange(len(f)), I)))
        I = get_Imax(R[-1], c, max_iter=max_iter_GD)
        I_list.append(I)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    def sort_act(I_list, f):
        sorted = []
        act_gains = [gaussian_chirlet_transform(f, I) for I in I_list]
        while act_gains:
            j = np.argmax(act_gains)
            sorted.append(I_list[j])
            I_list.pop(j)
            act_gains.pop(j)

        return sorted

    return sort_act(I_list, f), time.time() - start_time

def ACT_opt(f, P, max_iter_GD =200):
    # This function is exactly the same as ACT but slightly optimized
    N = len(f)
    i0 = 1
    a = 2
    gamma = np.arange(N+1)
    T = N
    F = 2*np.pi
    D = int(np.log(N)/np.log(a))
    k = np.arange(int(0.5*D)-i0+1)
    m_k = 4*np.power(a, 2*k)-1
    m = [np.arange(mk) for mk in m_k]

    wc = (2*np.pi/N)*gamma
    c = [F*m[-1][int(i/4)]*(i%4 + 1)/(T*40*np.power(a, 2*k[-1])) for i in range(4*m_k[-1])]
    dt = np.power(a, np.arange(D))

    def gaussian_chirplet(t, I):
        tc, wc, c, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*(c*(t-tc)+wc)*(t-tc)
        )

    def gaussian_chirlet_transform(f, I):
        return np.abs(np.vdot(f, gaussian_chirplet(np.arange(len(f)), I)))

    def get_Imax(Rnf, c, max_iter = 5000, alpha = 3e-3):

        def first_derivate_g(g, I):
            tc, wc, c, dt = I
            t = np.arange(len(g))
            dtc = ((t-tc)/(dt*dt) - 2.j*c*(t-tc) -1.j*wc)*g
            dwc = 1.j*(t-tc)*g

            return dtc, dwc

        def f_prime(Rnf, g, I):
            dtc, dwc = first_derivate_g(g, I)

            sc = np.vdot(Rnf, g)

            scdtc = np.vdot(Rnf, dtc)
            dfdtc = scdtc*np.conj(sc) + sc*np.conj(scdtc)

            scdwc = np.vdot(Rnf, dwc)
            dfdwc = scdwc*np.conj(sc) + sc*np.conj(scdwc)

            return 0.5*dfdtc/np.abs(sc), 0.5*dfdwc/np.abs(sc)

        N = len(Rnf)
        I = np.argmax(Rnf), np.pi, 0, a**D
        tc_, wc_, c_, dt_, = I

        # Find the best center frequency in the dictionnary with chirpiness c=0 and maximum window size

        def search_best_w(f, I, wc):
            N = len(f)
            tc_, wc_, c_, dt_, = I
            Id = np.ones(N)
            t = np.outer(Id, np.arange(N))
            w = np.outer(wc[:-1], Id)

            chirplets = (1/np.power(np.abs(dt_)*np.power(np.pi,0.5),0.5))*np.exp(
                -0.5*np.power((t-tc_)/dt_,2) + 1.j*(c_*(t-tc_)+w)*(t-tc_)
            )

            scalar_products = np.abs(np.sum(np.outer(Id, f) * np.conj(chirplets), axis=1))
            wcmax = wc[np.argmax(scalar_products)]

            return wcmax

        def search_best_c(f, I, c):
            N = len(f)
            M = len(c)
            tc_, wc_, c_, dt_, = I
            IdN = np.ones(N)
            IdM = np.ones(M)
            t = np.outer(IdM, np.arange(N))
            c_mat = np.outer(c, IdN)

            chirplets = (1/np.power(np.abs(dt_)*np.power(np.pi,0.5),0.5))*np.exp(
                -0.5*np.power((t-tc_)/dt_,2) + 1.j*(c_mat*(t-tc_)+wc_)*(t-tc_)
            )

            scalar_products = np.abs(np.sum(np.outer(IdM, f) * np.conj(chirplets), axis=1))
            cmax = c[np.argmax(scalar_products)]

            return cmax

        wc_ = search_best_w(Rnf, I, wc)
        I = tc_, wc_, c_, dt_

        # Refine tc and wc with chirpiness =0 and maximum window size

        g = [gaussian_chirplet(t, I) for t in range(len(f))]
        dtc, dwc = f_prime(Rnf, g, I)
        dtc, dwc = np.real(dtc), np.real(dwc)
        vtc, vwc = alpha * dtc, alpha * dwc
        nb_iter = 0

        while (np.abs(vtc) >= 1e-4 or N*np.abs(vwc)/(2*np.pi) >= 1e-4 or nb_iter%1000 in [0,1]) and nb_iter<max_iter:
            dtc, dwc = f_prime(Rnf, g, I)
            dtc, dwc =  np.real(dtc), np.real(dwc)
            vtc, vwc = 0.9*vtc + alpha * dtc, 0.9*vwc + alpha * dwc
            if tc_ + vtc < 0 or tc_ + vtc > N:
                vtc = 0
            if wc_ + vwc< 0 or wc_ + vwc > 2*np.pi:
                vwc = 0
            tc_, wc_ = tc_ + vtc, wc_ + vwc

            I = tc_, wc_, c_, dt_
            g = [gaussian_chirplet(t, I) for t in range(len(f))]
            nb_iter += 1

        # Find the best chirpiness in the dictinary with maximum window size

        c_ = search_best_c(f, I, c)
        I = tc_, wc_, c_, dt_

        scmax = gaussian_chirlet_transform(Rnf, I)
        #dtmax = dt_

        #for dt_ in dt:
        #    I = tc_, wc_, c_, dt_
        #    sc = gaussian_chirlet_transform(Rnf, I)
        #    if sc > scmax:
        #        dtmax = dt_
        #        scmax = sc

        #I = tc_, wc_, c_, dtmax

        if scmax != scmax:
            I = 0, 0, 0, dt_

        return I

    start_time = time.time()

    R = [f]
    I = get_Imax(R[-1], c, max_iter=max_iter_GD)
    I_list = [I]

    for n in range(1, P):
        R.append(R[-1] - np.multiply(gaussian_chirlet_transform(f, I), gaussian_chirplet(np.arange(len(f)), I)))
        I = get_Imax(R[-1], c, max_iter=max_iter_GD)
        I_list.append(I)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    def sort_act(I_list, f):
        sorted = []
        act_gains = [gaussian_chirlet_transform(f, I) for I in I_list]
        while act_gains:
            j = np.argmax(act_gains)
            sorted.append(I_list[j])
            I_list.pop(j)
            act_gains.pop(j)

        return sorted

    return sort_act(I_list, f), time.time() - start_time

def ACT_greedy(f, P, max_iter_GD =200):
    # This functions alternats between searching wc and c until reaching steady state
    N = len(f)
    i0 = 1
    a = 2
    gamma = np.arange(N+1)
    T = N
    F = 2*np.pi
    D = int(np.log(N)/np.log(a))
    k = np.arange(int(0.5*D)-i0+1)
    m_k = 4*np.power(a, 2*k)-1
    m = [np.arange(mk) for mk in m_k]

    wc = (2*np.pi/N)*gamma
    c = [F*m[-1][i]/(T*10*np.power(a, 2*k[-1])) for i in range(m_k[-1])]
    dt = np.power(a, np.arange(D))

    def gaussian_chirplet(t, I):
        tc, wc, c, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*(c*(t-tc)+wc)*(t-tc)
        )

    def gaussian_chirlet_transform(f, I):
        return np.abs(np.vdot(f, gaussian_chirplet(np.arange(len(f)), I)))

    def get_Imax(Rnf, c):
        N = len(Rnf)
        I = np.argmax(Rnf), np.pi, 0, a**D
        tc_, wc_, c_, dt_, = I

        # Find the best center frequency in the dictionnary with chirpiness c=0 and maximum window size

        def search_best_w(f, I, wc):
            N = len(f)
            tc_, wc_, c_, dt_, = I
            Id = np.ones(N)
            t = np.outer(Id, np.arange(N))
            w = np.outer(wc[:-1], Id)

            chirplets = (1/np.power(np.abs(dt_)*np.power(np.pi,0.5),0.5))*np.exp(
                -0.5*np.power((t-tc_)/dt_,2) + 1.j*(c_*(t-tc_)+w)*(t-tc_)
            )

            scalar_products = np.abs(np.sum(np.outer(Id, f) * np.conj(chirplets), axis=1))
            wcmax = wc[np.argmax(scalar_products)]

            return wcmax

        def search_best_c(f, I, c):
            N = len(f)
            M = len(c)
            tc_, wc_, c_, dt_, = I
            IdN = np.ones(N)
            IdM = np.ones(M)
            t = np.outer(IdM, np.arange(N))
            c_mat = np.outer(c, IdN)

            chirplets = (1/np.power(np.abs(dt_)*np.power(np.pi,0.5),0.5))*np.exp(
                -0.5*np.power((t-tc_)/dt_,2) + 1.j*(c_mat*(t-tc_)+wc_)*(t-tc_)
            )

            scalar_products = np.abs(np.sum(np.outer(IdM, f) * np.conj(chirplets), axis=1))
            cmax = c[np.argmax(scalar_products)]

            return cmax

        sc = gaussian_chirlet_transform(Rnf, I)
        delta_sc = sc
        i = 0
        while (delta_sc > 1e-3 and i<10):
            wc_ = search_best_w(Rnf, I, wc)
            I = tc_, wc_, c_, dt_
            c_ = search_best_c(Rnf, I, c)
            I = tc_, wc_, c_, dt_
            new_sc = gaussian_chirlet_transform(Rnf, I)
            delta_sc = np.abs(sc - new_sc)
            sc = new_sc
            i+=1
            #print(i, sc)

        scmax = gaussian_chirlet_transform(Rnf, I)

        if scmax != scmax:
            I = 0, 0, 0, dt_

        return I

    start_time = time.time()

    R = [f]
    I = get_Imax(R[-1], c)
    I_list = [I]

    for n in range(1, P):
        R.append(R[-1] - np.multiply(gaussian_chirlet_transform(f, I), gaussian_chirplet(np.arange(len(f)), I)))
        I = get_Imax(R[-1], c)
        I_list.append(I)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    def sort_act(I_list, f):
        sorted = []
        act_gains = [gaussian_chirlet_transform(f, I) for I in I_list]
        while act_gains:
            j = np.argmax(act_gains)
            sorted.append(I_list[j])
            I_list.pop(j)
            act_gains.pop(j)

        return sorted

    return sort_act(I_list, f), time.time() - start_time

def ACT_exhaustive(f, P):
    # This algorithm does an exhaustive search of c and wc (and does not refine with gradient descent)
    N = len(f)
    i0 = 1
    a = 2
    gamma = np.arange(N+1)
    T = N
    F = 2*np.pi
    D = int(np.log(N)/np.log(a))
    k = np.arange(int(0.5*D)-i0+1)
    m_k = 4*np.power(a, 2*k)-1
    m = [np.arange(mk) for mk in m_k]

    wc = (2*np.pi/N)*gamma
    c = [F*m[-1][i]/(T*10*np.power(a, 2*k[-1])) for i in range(m_k[-1])]
    dt = np.power(a, np.arange(D))

    def gaussian_chirplet(t, I):
        tc, wc, c, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*(c*(t-tc)+wc)*(t-tc)
        )

    def gaussian_chirlet_transform(f, I):
        return np.abs(np.vdot(f, gaussian_chirplet(np.arange(len(f)), I)))

    def get_Imax(Rnf, c):
        I = np.argmax(Rnf), np.pi, 0, a**D
        tc_, wc_, c_, dt_, = I

        # Find the best center frequency in the dictionnary with chirpiness c=0 and maximum window size

        def search_best_w_c(f, I, wc, c):
            N = len(f)
            M = len(c)
            tc_, wc_, c_, dt_, = I
            Id = np.ones(N)
            t = np.multiply.outer(Id, np.arange(N))
            w = np.outer(wc[:-1], Id)

            scalar_products = np.zeros((M, N))
            for i in range(len(c)):
                c_ = c[i]
                chirplets = (1/np.power(np.abs(dt_)*np.power(np.pi,0.5),0.5))*np.exp(
                    -0.5*np.power((t-tc_)/dt_,2) + 1.j*(c_*(t-tc_)+w)*(t-tc_)
                )

                scalar_products[i] = np.abs(np.sum(np.outer(Id, f) * np.conj(chirplets), axis=1))

            imax = np.argmax(scalar_products)
            wcmax = wc[imax%N]
            cmax = c[imax//N]

            #print(wcmax, cmax, scalar_products[imax//N][imax%N])

            return wcmax, cmax

        wc_, c_ = search_best_w_c(Rnf, I, wc, c)
        I = tc_, wc_, c_, dt_

        scmax = gaussian_chirlet_transform(Rnf, I)

        if scmax != scmax:
            I = 0, 0, 0, dt_

        return I

    start_time = time.time()

    R = [f]
    I = get_Imax(R[-1], c)
    I_list = [I]

    for n in range(1, P):
        R.append(R[-1] - np.multiply(gaussian_chirlet_transform(f, I), gaussian_chirplet(np.arange(len(f)), I)))
        I = get_Imax(R[-1], c)
        I_list.append(I)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    def sort_act(I_list, f):
        sorted = []
        act_gains = [gaussian_chirlet_transform(f, I) for I in I_list]
        while act_gains:
            j = np.argmax(act_gains)
            sorted.append(I_list[j])
            I_list.pop(j)
            act_gains.pop(j)

        return sorted

    return sort_act(I_list, f), time.time() - start_time

def AWT(f, P, max_iter_GD =200):
    # This function is exactly the same as ACT but was used to show the evolution of the results (it just has lots of prints)
    N = len(f)
    i0 = 1
    a = 2
    gamma = np.arange(N+1)
    T = N
    F = 2*np.pi
    D = int(np.log(N)/np.log(a))
    k = np.arange(int(0.5*D)-i0+1)
    m_k = np.array([4*np.power(a, 2*k_)-1 for k_ in k])
    m = [np.arange(mk) for mk in m_k]

    wc = (2*np.pi/N)*gamma
    dt = [np.power(a, k_) for k_ in range(D)]

    def gaussian_chirplet(t, I):
        tc, wc, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*wc*(t-tc)
        )

    def gaussian_chirlet_transform(f, I):
        return np.abs(np.vdot(f, gaussian_chirplet(np.arange(len(f)), I)))

    def get_Imax(Rnf, max_iter = 5000, alpha = 3e-3):

        def first_derivate_g(g, I):
            tc, wc, dt = I
            t = np.arange(len(g))
            dtc = ((t-tc)/(dt*dt) -1.j*wc)*g
            dwc = 1.j*(t-tc)*g

            return dtc, dwc

        def f_prime(Rnf, g, I):
            dtc, dwc = first_derivate_g(g, I)

            sc = np.vdot(Rnf, g)

            scdtc = np.vdot(Rnf, dtc)
            dfdtc = scdtc*np.conj(sc) + sc*np.conj(scdtc)

            scdwc = np.vdot(Rnf, dwc)
            dfdwc = scdwc*np.conj(sc) + sc*np.conj(scdwc)

            return 0.5*dfdtc/np.abs(sc), 0.5*dfdwc/np.abs(sc)

        N = len(Rnf)
        I = np.argmax(Rnf), np.pi, a**D
        tc_, wc_, dt_, = I

        # Find the best center frequency in the dictionnary with chirpiness c=0 and maximum window size

        wcmax = I[1]
        scmax = gaussian_chirlet_transform(Rnf, I)

        for wc_ in wc:
            I = tc_, wc_, dt_
            sc = gaussian_chirlet_transform(Rnf, I)
            if sc > scmax:
                scmax = sc
                wcmax = wc_

        wc_ = wcmax
        I = tc_, wc_, dt_

        # Refine tc and wc with chirpiness =0 and maximum window size

        g = [gaussian_chirplet(t, I) for t in range(len(f))]
        dtc, dwc = f_prime(Rnf, g, I)
        dtc, dwc = np.real(dtc), np.real(dwc)
        vtc, vwc = alpha * dtc, alpha * dwc
        nb_iter = 0

        while (np.abs(vtc) >= 1e-4 or N*np.abs(vwc)/(2*np.pi) >= 1e-4 or nb_iter%1000 in [0,1]) and nb_iter<max_iter:
            dtc, dwc = f_prime(Rnf, g, I)
            dtc, dwc =  np.real(dtc), np.real(dwc)
            vtc, vwc = 0.9*vtc + alpha * dtc, 0.9*vwc + alpha * dwc
            if tc_ + vtc < 0 or tc_ + vtc > N:
                vtc = 0
            if wc_ + vwc< 0 or wc_ + vwc > 2*np.pi:
                vwc = 0
            tc_, wc_ = tc_ + vtc, wc_ + vwc

            I = tc_, wc_, dt_
            g = [gaussian_chirplet(t, I) for t in range(len(f))]
            nb_iter += 1

        scmax = gaussian_chirlet_transform(Rnf, I)
        dtmax = dt_

        for dt_ in dt:
            I = tc_, wc_, dt_
            sc = gaussian_chirlet_transform(Rnf, I)
            if sc > scmax:
                dtmax = dt_
                scmax = sc

        I = tc_, wc_, dtmax

        if scmax != scmax:
            I = 0, 0, dt_

        return I

    start_time = time.time()

    R = [f]
    I = get_Imax(R[-1], max_iter=max_iter_GD)
    I_list = [I]

    for n in range(1, P):
        R.append(R[-1] - np.multiply(gaussian_chirlet_transform(f, I), gaussian_chirplet(np.arange(len(f)), I)))
        I = get_Imax(R[-1], max_iter=max_iter_GD)
        I_list.append(I)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    def sort_act(I_list, f):
        sorted = []
        act_gains = [gaussian_chirlet_transform(f, I) for I in I_list]
        while act_gains:
            j = np.argmax(act_gains)
            sorted.append(I_list[j])
            I_list.pop(j)
            act_gains.pop(j)

        return sorted

    return sort_act(I_list, f)

def AWT_(f, P, max_iter_GD =500):
    # This function is exactly the same as ACT but was used to show the evolution of the results (it just has lots of prints)
    N = len(f)
    i0 = 1
    a = 2
    gamma = np.arange(N+1)
    T = N
    F = 2*np.pi
    D = int(np.log(N)/np.log(a))
    k = np.arange(int(0.5*D)-i0+1)
    m_k = np.array([4*np.power(a, 2*k_)-1 for k_ in k])
    m = [np.arange(mk) for mk in m_k]

    wc = (2*np.pi/N)*gamma
    dt = [np.power(a, k_) for k_ in range(D)]

    def gaussian_chirplet(t, I):
        tc, wc, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*wc*(t-tc)
        )

    def scalar_product(f,g):
        return sum([f[t]*np.conj(g[t]) for t in range(len(f))])

    def gaussian_chirlet_transform(f, I):
        return np.abs(scalar_product(f, [gaussian_chirplet(t, I) for t in range(len(f))]))

    def get_Imax(Rnf, dt, max_iter = 5000, alpha = 3e-3):

        def first_derivate_g(g, I):
            tc, wc, dt = I
            dtc = [((t-tc)/(dt*dt) -1.j*wc)*g[t] for t in range(len(g))]
            dwc = [1.j*(t-tc)*g[t] for t in range(len(g))]

            return dtc, dwc

        def f_prime(Rnf, g, I):
            dtc, dwc = first_derivate_g(g, I)

            sc = scalar_product(Rnf, g)

            scdtc = scalar_product(Rnf, dtc)
            dfdtc = scdtc*np.conj(sc) + sc*np.conj(scdtc)

            scdwc = scalar_product(Rnf, dwc)
            dfdwc = scdwc*np.conj(sc) + sc*np.conj(scdwc)

            return 0.5*dfdtc/np.abs(sc), 0.5*dfdwc/np.abs(sc)

        N = len(Rnf)
        I = np.argmax(Rnf), np.pi, a**D
        tc_, wc_, dt_, = I
        g = [gaussian_chirplet(t, I) for t in range(len(f))]

        # Find the best center frequency in the dictionnary with chirpiness c=0 and maximum window size

        wcmax = I[1]
        scmax = np.abs(scalar_product(Rnf, g))

        for wc_ in wc:
            I = tc_, wc_, dt_
            g = [gaussian_chirplet(t, I) for t in range(len(f))]
            sc = np.abs(scalar_product(Rnf, g))
            if sc > scmax:
                scmax = sc
                wcmax = wc_

        wc_ = wcmax
        I = tc_, wc_, dt_

        # Refine tc and wc with chirpiness =0 and maximum window size

        g = [gaussian_chirplet(t, I) for t in range(len(f))]
        dtc, dwc = f_prime(Rnf, g, I)
        dtc, dwc = np.real(dtc), np.real(dwc)
        vtc, vwc = alpha * dtc, alpha * dwc
        nb_iter = 0

        while (np.abs(vtc) >= 1e-4 or N*np.abs(vwc)/(2*np.pi) >= 1e-4 or nb_iter%1000 in [0,1]) and nb_iter<max_iter:
            dtc, dwc = f_prime(Rnf, g, I)
            dtc, dwc =  np.real(dtc), np.real(dwc)
            vtc, vwc = 0.9*vtc + alpha * dtc, 0.9*vwc + alpha * dwc
            if tc_ + vtc < 0 or tc_ + vtc > N:
                vtc = 0
            if wc_ + vwc< 0 or wc_ + vwc > 2*np.pi:
                vwc = 0
            tc_, wc_ = tc_ + vtc, wc_ + vwc

            I = tc_, wc_, dt_
            g = [gaussian_chirplet(t, I) for t in range(len(f))]
            nb_iter += 1

        I = tc_, wc_, dt_
        g = [gaussian_chirplet(t, I) for t in range(len(f))]

        scmax = np.abs(scalar_product(Rnf,g))
        dtmax = dt_

        for dt_ in dt:
            I = tc_, wc_, dt_
            g = [gaussian_chirplet(t, I) for t in range(len(f))]
            sc = np.abs(scalar_product(Rnf,g))
            if sc > scmax:
                dtmax = dt_
                scmax = sc

        I = tc_, wc_, dtmax

        if scmax != scmax:
            I = 0, 0, 0, dt_

        return I

    start_time = time.time()

    R = [f]
    I = get_Imax(R[-1], dt, max_iter=max_iter_GD)
    I_list = [I]

    for n in range(1, P):
        R.append(R[-1]-np.multiply(gaussian_chirlet_transform(f, I),[gaussian_chirplet(t, I) for t in range(len(f))]))
        I = get_Imax(R[-1], dt, max_iter=max_iter_GD)
        I_list.append(I)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    def sort_act(I_list, f):
        sorted = []
        act_gains = [gaussian_chirlet_transform(f, I) for I in I_list]
        while act_gains:
            j = np.argmax(act_gains)
            sorted.append(I_list[j])
            I_list.pop(j)
            act_gains.pop(j)

        return sorted

    return sort_act(I_list, f)

def ACT_MP(f, P, max_iter_GD =1000):
    # This function is exactly the same as ACT but was used to show the evolution of the results (it just has lots of prints)
    N = len(f)
    i0 = 1
    a = 2
    gamma = np.arange(N+1)
    T = N
    F = 2*np.pi
    D = int(np.log(N)/np.log(a))
    k = np.arange(int(0.5*D)-i0+1)
    m_k = np.array([4*np.power(a, 2*k_)-1 for k_ in k])
    m = [np.arange(mk) for mk in m_k]

    tc = gamma
    wc = (2*np.pi/N)*gamma
    c = [F*m[-1][int(i/4)]*(i%4 + 1)/(T*4*np.power(a, 2*k[-1])) for i in range(4*m_k[-1])]
    dt = [np.power(a, k_) for k_ in range(D)]

    def gaussian_chirplet(t, I):
        tc, wc, c, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*(c*(t-tc)+wc)*(t-tc)
        )

    def gaussian_chirlet_transform(f, I):
        return np.abs(np.vdot(f, gaussian_chirplet(np.arange(len(f)), I)))

    def WVgI(I, N, k=4):
        tc, wc, c, dt = I
        Id = np.ones(k*N)
        t = np.outer(Id, np.arange(k*N))
        w = np.outer(np.arange(k*N), Id)
        return 2*np.exp(-np.power((t/k-tc)/dt, 2) - np.power(dt*((2*np.pi*w/N)/k - wc - 2*c*(t/k-tc)), 2))


    def get_Imax(Rnf, c, dt, max_iter = 5000, alpha = 3e-3):

        def first_derivate_g(g, I):
            tc, wc, c, dt = I
            t = np.arange(len(g))
            dtc = ((t-tc)/(dt*dt) - 2.j*c*(t-tc) -1.j*wc)*g
            dwc = 1.j*(t-tc)*g

            return dtc, dwc

        def f_prime(Rnf, g, I):
            dtc, dwc = first_derivate_g(g, I)

            sc = np.vdot(Rnf, g)

            scdtc = np.vdot(Rnf, dtc)
            dfdtc = scdtc*np.conj(sc) + sc*np.conj(scdtc)

            scdwc = np.vdot(Rnf, dwc)
            dfdwc = scdwc*np.conj(sc) + sc*np.conj(scdwc)

            return 0.5*dfdtc/np.abs(sc), 0.5*dfdwc/np.abs(sc)

        N = len(Rnf)
        I = np.argmax(Rnf), np.pi, 0, a**D
        tc_, wc_, c_, dt_, = I

        # Find the best center frequency in the dictionnary with chirpiness c=0 and maximum window size

        wcmax = I[1]
        scmax = gaussian_chirlet_transform(Rnf, I)

        for wc_ in wc:
            I = tc_, wc_, c_, dt_
            sc = gaussian_chirlet_transform(Rnf, I)
            if sc > scmax:
                scmax = sc
                wcmax = wc_

        print("wc :", wcmax)

        wc_ = wcmax
        I = tc_, wc_, c_, dt_

        # Refine tc and wc with chirpiness =0 and maximum window size

        g = [gaussian_chirplet(t, I) for t in range(len(f))]
        dtc, dwc = f_prime(Rnf, g, I)
        dtc, dwc = np.real(dtc), np.real(dwc)
        vtc, vwc = alpha * dtc, alpha * dwc
        nb_iter = 0

        tcl = [tc_]
        wcl = [wc_]

        while (np.abs(vtc) >= 1e-4 or N*np.abs(vwc)/(2*np.pi) >= 1e-4 or nb_iter%1000 in [0,1]) and nb_iter<max_iter:
            dtc, dwc = f_prime(Rnf, g, I)
            dtc, dwc =  np.real(dtc), np.real(dwc)
            vtc, vwc = 0.9*vtc + alpha * dtc, 0.9*vwc + alpha * dwc
            if tc_ + vtc < 0 or tc_ + vtc > N:
                vtc = 0
            if wc_ + vwc< 0 or wc_ + vwc > 2*np.pi:
                vwc = 0
            tc_, wc_ = tc_ + vtc, wc_ + vwc

            tcl.append(tc_)
            wcl.append(wc_)

            I = tc_, wc_, c_, dt_
            g = [gaussian_chirplet(t, I) for t in range(len(f))]
            nb_iter += 1

        # Find the best chirpiness in the dictinary with maximum window size

        cmax = I[2]
        scmax = gaussian_chirlet_transform(Rnf, I)

        for c_ in c:
            I = tc_, wc_, c_, dt_
            sc = gaussian_chirlet_transform(Rnf, I)
            if sc > scmax:
                scmax = sc
                cmax = c_

        print("c :", cmax)

        c_ = cmax
        I = tc_, wc_, c_, dt_

        scmax = gaussian_chirlet_transform(Rnf, I)
        dtmax = dt_

        for dt_ in dt:
            I = tc_, wc_, c_, dt_
            sc = gaussian_chirlet_transform(Rnf, I)
            if sc > scmax:
                dtmax = dt_
                scmax = sc

        I = tc_, wc_, c_, dtmax
        #g = [gaussian_chirplet(t, I) for t in range(len(f))]

        #plt.plot(np.arange(len(g)), np.real(np.array(g)))
        #plt.show()

        if scmax != scmax:
            I = 0, 0, 0, dt_
            scmax = gaussian_chirlet_transform(Rnf, I)

        print("scalar product :",scmax)

        return I

    start_time = time.time()

    R = [f]
    I = get_Imax(R[-1], c, dt, max_iter=max_iter_GD)
    I_list = [I]
    print("I :", I)
    print("step 1")

    sum_ = np.power(gaussian_chirlet_transform(f, I), 2)*WVgI(I,len(f))

    for n in range(1, P):
        R.append(R[-1]-gaussian_chirlet_transform(f, I)*gaussian_chirplet(np.arange(len(f)), I))
        I = get_Imax(R[-1], c, dt, max_iter=max_iter_GD)
        I_list.append(I)
        #plt.plot(np.linspace(0, len(f), len(f)), [np.real(R[-1][t]) for t in range(len(f))])
        #plt.show()
        print("I :",I)
        print("step " + str(n+1))

        sum_ = np.add(sum_, np.power(gaussian_chirlet_transform(f, I), 2)*WVgI(I,len(f)))

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    def sort_act(I_list, f):
        sorted = []
        act_gains = [gaussian_chirlet_transform(f, I) for I in I_list]
        while act_gains:
            j = np.argmax(act_gains)
            sorted.append(I_list[j])
            I_list.pop(j)
            act_gains.pop(j)

        return sorted

    return sum_, sort_act(I_list, f)

def ACT_MP_LEM(f, P, threshold1 = 1e-3, threshold2 = 1e-12, iter_max1 = 200, iter_max2 = 10):
    N = len(f)
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

    def gaussian_chirplet(t, I):
        tc, wc, c, dt = I
        return (1/np.power(np.abs(dt)*np.power(np.pi,0.5),0.5))*np.exp(
            -0.5*np.power((t-tc)/dt,2) + 1.j*(c*(t-tc)+wc)*(t-tc)
        )

    def norm_l2(fct):
        return np.power(np.abs(np.vdot(fct, fct)), 0.5)

    def gaussian_chirlet_transform(f, I):
        return np.abs(np.vdot(f, [gaussian_chirplet(t, I) for t in range(len(f))]))

    def WVgI(I, N, k=4):
        tc, wc, c, dt = I
        Id = np.ones(k*N)
        t = np.outer(Id, np.arange(k*N))
        w = np.outer(np.arange(k*N), Id)
        return 2*np.exp(-np.power((t/k-tc)/dt, 2) - np.power(dt*((2*np.pi*w/N)/k - wc - 2*c*(t/k-tc)), 2))

    def kernel(dt, c, t):
        return np.exp(-0.5*np.power(t/dt, 2) + 1j*c*t*t)/np.power(np.power(np.pi, 0.5)*dt, 0.5)

    def get_Imax(Rnf, c, dt, max_iter = 200, alpha = 1):

        def first_derivate_g(g, I):
            tc, wc, c, dt = I
            t = np.arange(len(g))
            dtc = ((t-tc)/(dt*dt) - 2.j*c*(t-tc) -1.j*wc)*g
            dwc = 1.j*(t-tc)*g

            return dtc, dwc

        def f_prime(Rnf, g, I):
            dtc, dwc = first_derivate_g(g, I)

            sc = np.vdot(Rnf, g)

            scdtc = np.vdot(Rnf, dtc)
            dfdtc = scdtc*np.conj(sc) + sc*np.conj(scdtc)

            scdwc = np.vdot(Rnf, dwc)
            dfdwc = scdwc*np.conj(sc) + sc*np.conj(scdwc)

            return 0.5*dfdtc/np.abs(sc), 0.5*dfdwc/np.abs(sc)

        N = len(Rnf)
        cmax = 0
        dtmax = 1
        scmax = 0
        wcmax = 0
        tc_ = np.argmax(Rnf)
        #wc_ = np.pi
        for wc_ in 2*np.pi*np.arange(10)/10:
            for k in range(len(dt)):
                dt_ = dt[k]
                for c_ in c[k]:
                    t = np.arange(N)
                    ker = kernel(dt_, c_, t-tc_)*np.exp(-1j*t*wc_)
                    sc = np.abs(np.dot(Rnf, ker))
                    if sc>scmax:
                        cmax = c_
                        dtmax = dt_
                        scmax = sc
                        wcmax = wc_
        c_, dt_, wc_ = cmax, dtmax, wcmax
        I = tc_, wc_, c_, dt_
        g = [gaussian_chirplet(t, I) for t in range(len(f))]
        dtc, dwc = f_prime(Rnf, g, I)
        dtc, dwc = np.real(dtc), np.real(dwc)
        vtc, vwc = 0.1*alpha * dtc, alpha * dwc
        nb_iter = 0
        while (np.abs(vtc) >= 1e-3 or N*np.abs(vwc)/(2*np.pi) >= 1e-3 or nb_iter == 0) and nb_iter<max_iter:
            dtc, dwc = f_prime(Rnf, g, I)
            dtc, dwc = np.real(dtc), np.real(dwc)
            vtc, vwc = 0.9*vtc + 0.1*alpha * dtc, 0.9*vwc + alpha * dwc
            if tc_ + vtc < 0 or tc_ + vtc > N:
                vtc = 0
            if wc_ + 2*np.pi*N*vwc/(2*np.pi)/N < 0 or wc_ + 2*np.pi*N*vwc/(2*np.pi)/N > 2*np.pi:
                vwc = 0
            tc_, wc_ = tc_ + vtc, wc_ + 2*np.pi*N*vwc/(2*np.pi)/N
            I = tc_, wc_, c_, dt_
            g = [gaussian_chirplet(t, I) for t in range(len(f))]
            nb_iter += 1
            sc = gaussian_chirlet_transform(Rnf, I)
            if sc>scmax:
                scmax = sc
        for k in range(len(dt)):
            dt_ = dt[k]
            for c_ in c[k]:
                t = np.arange(N)
                ker = kernel(dt_, c_, t-tc_)*np.exp(-1j*t*wc_)
                sc = np.abs(np.dot(Rnf, ker))
                if sc > scmax:
                    cmax = c_
                    dtmax = dt_
                    scmax = sc
        Imax = tc_, wc_, cmax, dtmax
        return Imax

    start_time = time.time()

    R = [f]
    I = get_Imax(R[-1], c, dt, max_iter=iter_max1)
    I_list = [I]
    p = 1
    cc = np.power(gaussian_chirlet_transform(f, I)/norm_l2(R[-1]),2)
    print("cc :",cc)
    print("I :",I)
    print("step 1")
    while cc > threshold2 and p < P:
        R.append(R[-1] - gaussian_chirlet_transform(f, I_list[-1])*gaussian_chirplet(np.arange(len(f)), I_list[-1]))
        I = get_Imax(R[-1], c, dt, max_iter=iter_max1)
        I_list.append(I)
        e = R[-1]
        nb_iter = 0
        I_min = I_list
        e_min = e
        norm_min = norm_l2(e)
        print("norm(e) before maximization :", norm_l2(e))
        while norm_l2(e) > threshold1 and nb_iter < iter_max2:
            new_I = []
            new_e = f
            for k in range(p+1):
                a_k = gaussian_chirlet_transform(f, I_list[k])
                g_k = gaussian_chirplet(np.arange(len(f)), I_list[k])
                y_k = a_k*g_k + 1/p*e
                I_k = get_Imax(y_k, c, dt, max_iter=iter_max1)
                new_I.append(I_k)
                new_e = new_e - gaussian_chirlet_transform(f, I_k) * gaussian_chirplet(np.arange(len(f)), I_k)
            e = new_e
            nb_iter += 1
            if norm_l2(e) < norm_min:
                I_min = new_I
                e_min = e
                norm_min = norm_l2(e)
        R[-1] = e_min
        I_list = I_min
        cc = np.power(gaussian_chirlet_transform(f, I_list[-1]) / norm_l2(R[-1]), 2)
        p+=1
        #plt.plot(np.linspace(0, len(f), len(f)), [np.real(R[-1][t]) for t in range(len(f))])
        #plt.show()
        print("norm(e) after maximization :", norm_l2(R[-1]))
        print("cc :",cc)
        print("I :", I_list[-1])
        print("step " + str(p))

    sum_ = np.power(np.abs(gaussian_chirlet_transform(f, I_list[0])), 2) * WVgI(I_list[0],len(f))
    for k in range(1, p):
        sum_ = sum_ + np.power(gaussian_chirlet_transform(f, I_list[k]), 2) * WVgI(I_list[k],len(f))

    print("time elapsed: {:.2f}s".format(time.time() - start_time))

    def sort_act(I_list, f):
        sorted = []
        act_gains = [gaussian_chirlet_transform(f, I) for I in I_list]
        while act_gains:
            j = np.argmax(act_gains)
            sorted.append(I_list[j])
            I_list.pop(j)
            act_gains.pop(j)

        return sorted

    return sum_, sort_act(I_list, f)