import logging
import os
import math
import numpy as np
import matplotlib.pyplot as plt

use_gpu = True
if use_gpu:
    import cupy as cp
    import cupyx.scipy.fft as fft
else:
    import numpy as cp
    import scipy.fft as fft

def around(x):
    return round(float(x))

def togpu(x):
    if use_gpu and not isinstance(x, cp.ndarray):
        return cp.array(x)
    else:
        return x


def tocpu(x):
    if use_gpu and isinstance(x, cp.ndarray):
        return x.get()
    else:
        return x

def mychirp(t, f0, f1, t1):
    betai = (f1 - f0) / t1
    phase = 2 * cp.pi * (f0 * t + 0.5 * betai * t * t)
    sig = cp.exp(1j * togpu(phase))
    return sig

class Config:

    sf = 12
    bw = 406250#*(1-20*1e-6)
    fs = 1e6
    sig_freq = 2.4e9
    # sig_freq = 2400000030.517578#-52e6/(2**18)
    preamble_len = 14#64  # TODO!!!!
    skip_preambles = 8  # skip first 8 preambles ## TODO
    total_len = 90.25-2#-16+64
    thresh = None# 0.03
    # file_paths = ['/data/djl/temp/OptimalPkt/fingerprint_data/data0_test_3',]
    cfo_range = bw // 8
    outfolder = "fout_test"

    wired = False
    if not wired:
        # base_path = '/data/djl/temp/OptimalPkt/fingerprint_data/'
        base_path = '/data/djl/OptimalPkt/fin'
        temp_list = []
        for filename in os.listdir(base_path):
            if 'data0' in filename:
                filename = os.path.join(base_path, filename)
                fileidx = filename.split('_')[-1]
                if fileidx.isnumeric():
                    if 'wavelen' in filename:
                        file_tuple = (filename, -int(fileidx)-1)
                    else:
                        file_tuple = (filename, int(fileidx))
                else:
                    file_tuple = (filename, -10)
                temp_list.append(file_tuple)

        file_paths_zip = sorted(temp_list, key=lambda pair: pair[1])

    else:
        # file_paths_zip = (("/data/djl/temp/OptimalPkt/hou2", 0),) # !!! TODO FOR DEBUG
        file_paths_zip = (("/data/djl/OptimalPkt/test_1218_4", 0),) # !!! TODO FOR DEBUG
        # file_paths_zip = (("/data/djl/OptimalPkt/fin/data0_test_200", 0),)  # !!! TODO FOR DEBUG
    if True:
        preamble_len = 512
        total_len = 90.25-16+512
        file_paths_zip = (("/data/djl/OptimalPkt/test_1218_3", 0),) # !!! TODO FOR DEBUG
    if True:
        preamble_len = 240
        total_len = 308.25
        file_paths_zip = (("/data/djl/OptimalPkt/test_1226", 0),) # !!! TODO FOR DEBUG


    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    nsamp = around(n_classes * fs / bw)
    nsampf = (n_classes * fs / bw)
    # f_lower, f_upper = -50000, -30000
    f_lower, f_upper = -38000, -34000
    t_lower, t_upper = 0, nsamp
    fguess = (f_lower + f_upper) / 2
    tguess = nsamp / 2
    code_len = 2

    # cfo_change_rate = 46/(60* n_classes * fs / bw) # Hz/sps

    tstandard = cp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    decode_matrix_a = cp.zeros((n_classes, nsamp), dtype=cp.complex64)
    decode_matrix_b = cp.zeros((n_classes, nsamp), dtype=cp.complex64)
    for code in range(n_classes):
        nsamples = around(nsamp / n_classes * (n_classes - code))
        refchirp = mychirp(tstandard, f0=bw * (-0.5 + code / n_classes), f1=bw * (0.5 + code / n_classes),
                           t1=2 ** sf / bw )
        decode_matrix_a[code, :nsamples] = cp.conj(refchirp[:nsamples])

        refchirp = mychirp(tstandard, f0=bw * (-1.5 + code / n_classes), f1=bw * (-0.5 + code / n_classes),
                           t1=2 ** sf / bw )
        decode_matrix_b[code, nsamples:] = cp.conj(refchirp[nsamples:])


    gen_refchirp_deadzone = 0
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)

    fft_upsamp = 1024
    detect_range_pkts = 6
    assert detect_range_pkts >= 2 # add 1, for buffer of cross-add
    detect_to_max = nsamp * 2
    fft_n = int(fs) #nsamp * fft_upsamp
    if use_gpu: plan = fft.get_fft_plan(cp.zeros(fft_n, dtype=cp.complex64))
    else: plan = None
    fft_ups = cp.zeros((preamble_len + detect_range_pkts, fft_n), dtype=cp.float32)
    fft_downs = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.float32)
    fft_ups_x = cp.zeros((preamble_len + detect_range_pkts, fft_n), dtype=cp.complex64)
    fft_downs_x = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.complex64)



logger = logging.getLogger('my_logger')
level = logging.WARNING
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(level)  # Set the console handler level
file_handler = logging.FileHandler('run_241115_2.log')
file_handler.setLevel(level)  # Set the file handler level
# formatter = logging.Formatter('%(message)s')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

if use_gpu:
    cp.cuda.Device(0).use()
Config = Config()



def add_freq(pktdata_in, est_cfo_freq):
    cfosymb = cp.exp(2j * cp.pi * est_cfo_freq * cp.linspace(0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
    cfosymb = cfosymb.astype(cp.complex64)
    pktdata2a = pktdata_in * cfosymb
    return pktdata2a




def average_modulus(lst, n_classes):
    complex_sum = cp.sum(cp.exp(1j * 2 * cp.pi * cp.array(lst) / n_classes))
    avg_angle = cp.angle(complex_sum)
    avg_modulus = (avg_angle / (2 * cp.pi)) * n_classes
    return avg_modulus



def gen_upchirp(t0, td, f0, betai):
    # start from ceil(t0in), end
    t = (cp.arange(math.ceil(t0), math.ceil(t0 + td), dtype=float) - t0)
    phase = 2 * cp.pi * (f0 * t + 0.5 * betai * t * t) / Config.fs
    sig = cp.exp(1j * phase)
    return sig

def converter_down(cfofreq, time_error):
    cfofreq -= (Config.f_upper + Config.f_lower) / 2
    cfofreq /= (Config.f_upper - Config.f_lower) / 2
    time_error -= (Config.t_upper + Config.t_lower) / 2
    time_error /= (Config.t_upper - Config.t_lower) / 2
    return cfofreq, time_error

def converter_up(cfofreq, time_error):
    cfofreq *= (Config.f_upper - Config.f_lower) / 2
    cfofreq += (Config.f_upper + Config.f_lower) / 2
    time_error *= (Config.t_upper - Config.t_lower) / 2
    time_error += (Config.t_upper + Config.t_lower) / 2
    return cfofreq, time_error



def cp_str(x, precision=2, suppress_small=False):
    return np.array2string(tocpu(x), precision=precision, formatter={'float_kind': lambda k: f"{k:.2f}"}, floatmode='fixed', suppress_small=suppress_small)


def myscatter(ax, x, y, **kwargs):
    ax.scatter(tocpu(x), tocpu(y), **kwargs)


def myplot(*args, **kwargs):
    if len(args) == 2:
        ax = args[0]
        ax.plot(tocpu(args[1]), **kwargs)
    elif len(args) == 3:
        ax = args[0]
        ax.plot(tocpu(args[1]), tocpu(args[2]), **kwargs)
    else:
        raise ValueError("plot function accepts either 1 or 2 positional arguments")




# freqs: before shift f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / n   if n is even
# after shift f = [-n/2, ..., -1, 0, 1, ...,   n/2-1] / n if n is even
# n is fft_n, f is cycles per sample spacing
# since fs=1e6: real freq in hz fhz=[-n/2, ..., -1, 0, 1, ...,   n/2-1] / n * 1e6Hz
# total range: sampling frequency. -fs/2 ~ fs/2, centered at 0
# bandwidth = 0.40625 sf
# fmax = tocpu(cp.argmax(cp.abs(data0))) - Config.fs / 2 # fftshift: -500000 to 500000 !!!
def myfft(chirp_data, n, plan):
    if use_gpu:
        return np.fft.fftshift(fft.fft(chirp_data.astype(cp.complex64), n=n, plan=plan))
    else:
        return np.fft.fftshift(fft.fft(chirp_data.astype(cp.complex64), n=n))



def dechirp_fft(tstart, fstart, pktdata_in, refchirp, pidx, ispreamble):
    nsymblen = 2 ** Config.sf / Config.bw * Config.fs# * (1 + fstart / Config.sig_freq)
    nstart = nsymblen * pidx + tstart
    start_pos_d = nstart - around(nstart)
    sig1 = pktdata_in[around(nstart): Config.nsamp + around(nstart)]
    if len(refchirp) != len(sig1):
        print(tstart, fstart, pidx, ispreamble)
        print(len(refchirp), len(sig1))
        plt.plot(tocpu(cp.unwrap(cp.angle(refchirp))))
        plt.plot(tocpu(cp.unwrap(cp.angle(sig1))))
        plt.show()

    sig2 = sig1 * refchirp
    freqdiff = start_pos_d / nsymblen * Config.bw / Config.fs * Config.fft_n
    if ispreamble: freqdiff -= fstart / Config.sig_freq * Config.bw * pidx
    else: freqdiff += fstart / Config.sig_freq * Config.bw * pidx
    sig2 = add_freq(sig2,freqdiff)
    data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
    # plt.plot(tocpu(cp.abs(data0)))
    # plt.show()
    return data0

def obj(xdata, ydata, coeff2d):
    return np.abs(ydata.dot(np.exp(-1j * np.polyval(coeff2d, xdata))))

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi