import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import griddata
from sobol import *

def chirp(t, tc, wc, c):
    return np.exp(1.j*(c*(t-tc)+wc)*(t-tc))

def obj(f, wc, c):
    g = np.imag(chirp(t, 0, wc, c))
    res = np.dot(f,g)
    return res

N = 256 # length of sample
t = np.arange(N)
M = 10 # number of parameters per sets
max_fqcy = 2*np.pi
c = np.arange(M)*max_fqcy/(2*N*M)
w0 = np.arange(M)*max_fqcy/(2*M)

dataset = np.zeros((M*M, N+2))
for m in range(M*M):
    c_m = c[m%M]
    w0_m = w0[m//M] # change w0 every M samples
    dataset[m][:-2] = np.sin((c_m*t+w0_m)*t) + np.random.normal(0, 0.05, N)
    dataset[m][-2] = c_m
    dataset[m][-1] = w0_m
clean_dataset = np.zeros((M*M, N+2))
for m in range(M*M):
    c_m = c[m%M]
    w0_m = w0[m//M] # change w0 every M samples
    clean_dataset[m][:-2] = np.sin((c_m*t+w0_m)*t)
    clean_dataset[m][-2] = c_m
    clean_dataset[m][-1] = w0_m

# We will sample the objective with Sobol sequence.
# Create the pseudo random set
samples = []
seed = 1234
for i in range(100000):
    seq, seed = i4_sobol(dim_num = 2, seed = seed)
    samples += [seq]
samples = np.array(samples)

try:
    os.makedirs('objective_landscape_viz')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for idx in range(len(dataset)):
    fig = plt.figure(figsize=(20,16))
    #plt.rcParams['font.size'] = 16
    fig.suptitle("Datum: idx = {}, c = {}, w_c = {}".format(idx, dataset[idx][-2], dataset[idx][-1]),y=1.02)
    spec = gridspec.GridSpec(ncols=6, nrows=6)
    top_left_ax = fig.add_subplot(spec[:2, :3])
    top_right_ax = fig.add_subplot(spec[:2, 3:])
    bottom_left_ax = fig.add_subplot(spec[2:, :3])
    bottom_right_ax = fig.add_subplot(spec[2:, 3:])

    f = dataset[idx][:-2]

    top_left_ax.plot(dataset[idx][:-2],label='Signal')
    top_left_ax.plot(np.imag(chirp(t,0,dataset[idx][-1],dataset[idx][-2])),label='Chirp')
    top_left_ax.legend()
    top_left_ax.set_title("Noisy data")

    _wc, _c = samples[:,0]*max_fqcy/2, samples[:,1]*max_fqcy/(2*N)
    z = [obj(f, _wc[i], _c[i]) for i in range(len(_wc))]
    wci, ci = np.meshgrid(np.linspace(0, max_fqcy/2, 1000), np.linspace(0, max_fqcy/(2*N), 1000))
    zi = griddata((_wc, _c), z, (wci, ci), method='nearest', rescale=True)
    pcm = bottom_left_ax.pcolormesh(wci, ci, zi, vmin=zi.min(), vmax=zi.max())
    bottom_left_ax.plot(dataset[idx][-1],dataset[idx][-2],'xr',markersize=12, markeredgewidth=4)
    bottom_left_ax.set_xlabel("$w_c$")
    bottom_left_ax.set_ylabel("$c$")
    bottom_left_ax.set_title("Scalar product (noisy data)")
    fig.colorbar(pcm, ax = bottom_left_ax, orientation="horizontal",pad=0.08)

    f = clean_dataset[idx][:-2]

    top_right_ax.plot(clean_dataset[idx][:-2],label='Signal')
    top_right_ax.plot(np.imag(chirp(t,0,clean_dataset[idx][-1],clean_dataset[idx][-2])),label='Chirp')
    top_right_ax.legend()
    top_right_ax.set_title("Clean data")

    _wc, _c = samples[:,0]*max_fqcy/2, samples[:,1]*max_fqcy/(2*N)
    z = [obj(f, _wc[i], _c[i]) for i in range(len(_wc))]
    wci, ci = np.meshgrid(np.linspace(0, max_fqcy/2, 1000), np.linspace(0, max_fqcy/(2*N), 1000))
    zi = griddata((_wc, _c), z, (wci, ci), method='nearest', rescale=True)
    pcm = bottom_right_ax.pcolormesh(wci, ci, zi, vmin=zi.min(), vmax=zi.max())
    bottom_right_ax.plot(clean_dataset[idx][-1],clean_dataset[idx][-2],'xr',markersize=12, markeredgewidth=4)
    bottom_right_ax.set_xlabel("$w_c$")
    bottom_right_ax.set_ylabel("$c$")
    bottom_right_ax.set_title("Scalar product (clean data)")
    fig.colorbar(pcm, ax = bottom_right_ax, orientation="horizontal",pad=0.08)
    fig.tight_layout()
    fig.savefig('./objective_landscape_viz/objective_landscape_viz_{:02d}.png'.format(idx),bbox_inches='tight')
    plt.close(fig)

