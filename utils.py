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

def mget(x):
    if isinstance(x, cp.ndarray) or isinstance(x, np.ndarray):
        assert x.shape == ()
        return x.item()
    else: return x
def sqlist(lst):
    return [item if isinstance(item, (int, float)) else item.item() for item in lst]
def tos(item):
    return item if isinstance(item, (int, float)) else item.item()

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
    beta = (f1 - f0) / t1
    phase = 2 * cp.pi * (f0 * t + 0.5 * beta * t * t)
    sig = cp.exp(1j * togpu(phase))
    return sig
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Sample argparse script")

# Add the integer argument with a default value of 7
parser.add_argument('--sf', type=int, default=7, help="Set the value of sf (default is 7)")
args = parser.parse_args()

class Config:
    sf = args.sf # parse hbq's 6~12 data
    bw = 406250#*(1-20*1e-6)
    sig_freq = 2.4e9
    preamble_len=15
    total_len = [136, 119, 107, 97, 90, 96, 89][sf - 6]
    file_paths_zip = (f"/data/djl/OptimalPkt/data0217/sf_{sf}_0116",)  # !!! TODO FOR DEBUG
    guess_f = -40000

    # sf = args.sf # parse outdoordata0217
    # bw = 125000
    # sig_freq = 470000000
    # preamble_len= 10
    # total_len = 42
    # file_paths_zip = []
    # for x in range(1, [5, 2, 2, 3, 3, 4][sf - 7]):
    #     file_paths_zip.append(f"/data/djl/datasets/outdoordata0217/sf{sf}{x}.sigdat")
    # outpath = f"/data/djl/datasets/outdoordata0217_cut/sf{sf}"
    # guess_f = 0

    # sf = args.sf # parse /data/djl/250125-unsplit
    # bw = 125000
    # sig_freq = 470000000
    # preamble_len= 10
    # total_len = 77
    # file_paths_zip = ['/data/djl/250125-unsplit/sf12_bw125k.sigdat', '/data/djl/250125-unsplit/sf12_bw125k_2.sigdat', '/data/djl/250125-unsplit/sf12_bw125k_3.sigdat','/data/djl/250125-unsplit/sf12_bw125k_4.sigdat']
    # # file_paths_zip = []
    # # for x in range(1, [5, 2, 2, 3, 3, 4][sf - 7]):
    # #     file_paths_zip.append(f"/data/djl/datasets/outdoordata0217/sf{sf}{x}.sigdat")
    # outpath = f"/data/djl/datasets/outdoordata0217_cut/sf{sf}"
    # guess_f = 0




    fs = 1e6
    skip_preambles = 8  # skip first 8 preambles ## TODO
    thresh = None# 0.03
    cfo_range = bw // 8
    code_len = 2
    outfolder = "fout"

    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    nsamp = around(n_classes * fs / bw)
    nsampf = (n_classes * fs / bw)

    # cfo_change_rate = 46/(60* n_classes * fs / bw) # Hz/sps

    tstandard = cp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    decode_matrix_a = cp.zeros((n_classes, nsamp), dtype=cp.complex64)
    decode_matrix_b = cp.zeros((n_classes, nsamp), dtype=cp.complex64)

    betai = bw / ((2 ** sf) / bw)
    for code in range(n_classes):
        nsamples = around(nsamp / n_classes * (n_classes - code))
        f01 = bw * (-0.5 + code / n_classes)
        refchirpc1 = cp.exp(-1j * 2 * cp.pi * (f01 * tstandard + 0.5 * betai * tstandard * tstandard))
        f02 = bw * (-1.5 + code / n_classes)
        refchirpc2 = cp.exp(-1j * 2 * cp.pi * (f02 * tstandard + 0.5 * betai * tstandard * tstandard))
        decode_matrix_a[code, :nsamples] = refchirpc1[:nsamples]
        if code > 0: decode_matrix_b[code, nsamples:] = refchirpc2[nsamples:]

    gen_refchirp_deadzone = 0
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)

    fft_upsamp = 1024
    detect_range_pkts = 4
    assert detect_range_pkts >= 2 # add 1, for buffer of cross-add
    detect_to_max = nsamp * 2
    fft_n = int(fs) #nsamp * fft_upsamp
    if use_gpu: plan = fft.get_fft_plan(cp.zeros(fft_n, dtype=cp.complex64))
    else: plan = None
    fft_ups = cp.zeros((preamble_len + detect_range_pkts, fft_n), dtype=cp.float32)
    fft_downs = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.float32)
    fft_ups_x = cp.zeros((preamble_len + detect_range_pkts, fft_n), dtype=cp.complex64)
    fft_downs_x = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.complex64)


logging.basicConfig(
    # format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    level=logging.WARNING
)

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('run_241219.log')
file_handler.setLevel(level=logging.DEBUG)  # Set the file handler level
# formatter = logging.Formatter('%(message)s')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

if use_gpu:
    cp.cuda.Device(0).use()
    logger.error("WARNING: USING GPU ")
else:
    logger.error("WARNING: NOT USING GPU ")
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



def gen_upchirp(t0, td, f0, beta):
    # start from ceil(t0in), end
    t = (cp.arange(math.ceil(t0), math.ceil(t0 + td), dtype=float) - t0)
    phase = 2 * cp.pi * (f0 * t + 0.5 * beta * t * t) / Config.fs
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
def myfft(chirp_data, n, plan):
    if use_gpu:
        return np.fft.fftshift(fft.fft(chirp_data.astype(cp.complex64), n=n, plan=plan))
    else:
        return np.fft.fftshift(fft.fft(chirp_data.astype(cp.complex64), n=n))



def dechirp_fft(tstart, fstart, pktdata_in, refchirp, pidx, ispreamble):
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs# * (1 + fstart / Config.sig_freq)
    start_pos_all = nsamp_small * pidx + tstart
    start_pos = around(start_pos_all)
    start_pos_d = start_pos_all - start_pos
    sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
    # plt.plot(tocpu(cp.unwrap(cp.angle(refchirp))))
    # plt.plot(tocpu(cp.unwrap(cp.angle(sig1))))
    # plt.show()
    sig2 = sig1 * refchirp
    freqdiff = start_pos_d / nsamp_small * Config.bw / Config.fs * Config.fft_n
    if ispreamble: freqdiff -= fstart / Config.sig_freq * Config.bw * pidx
    else: freqdiff += fstart / Config.sig_freq * Config.bw * pidx
    sig2 = add_freq(sig2,freqdiff)
    data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
    # plt.plot(tocpu(cp.abs(data0)))
    # plt.show()
    return data0

