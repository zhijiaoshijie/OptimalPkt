import argparse
import logging
import os
import random
import sys
import time
import itertools
from functools import partial
import pickle
import cmath
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# import pandas as pd
# Enable fallback mode
# from cupyx.fallback_mode import numpy as np
import numpy as np
import scipy.optimize as opt
import csv
# import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from tqdm import tqdm
# import scipy
import cupyx.scipy.fft as fft

logger = logging.getLogger('my_logger')
level = logging.INFO
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


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', default=False, help='Use cpu instead of gpu (numpy instead of cupy)')
parser.add_argument('--pltphase', action='store_true', default=False)
parser.add_argument('--searchphase', action='store_true', default=False)
parser.add_argument('--refine', action='store_true', default=False)
parser.add_argument('--searchphase_step', type=int, default=1000000)
parser.add_argument('--plotmap', action='store_true', default=False)
parser.add_argument('--plotline', action='store_true', default=False)
parser.add_argument('--end1', action='store_true', default=False)
parser.add_argument('--savefile', action='store_true', default=False)
parser.add_argument('--noplot', action='store_true', default=False)
parser.add_argument('--fromfile', type=str, default=None)
parser.add_argument('--compLTSNR', action='store_true', default=False)
parser.add_argument('--decode_unknown', action='store_true', default=False)
parser.add_argument('--snrlow', type=int, default=-20)
parser.add_argument('--snrhigh', type=int, default=-19)
parse_opts = parser.parse_args()

use_gpu = not parse_opts.cpu
if use_gpu:
    import cupy as cp
    import cupyx.scipy.fft as fft
else:
    import numpy as cp
    import scipy.fft as fft


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


def myfft(chirp_data, n, plan):
    return np.fft.fftshift(fft.fft(chirp_data.astype(cp.complex64), n=n, plan=plan))


script_path = __file__
mod_time = os.path.getmtime(script_path)
readable_time = time.ctime(mod_time)
logger.info(f"Last modified time of the script: {readable_time}")


def average_modulus(lst, n_classes):
    complex_sum = cp.sum(cp.exp(1j * 2 * cp.pi * cp.array(lst) / n_classes))
    avg_angle = cp.angle(complex_sum)
    avg_modulus = (avg_angle / (2 * cp.pi)) * n_classes
    return avg_modulus


class Config:
    sf = 12
    bw = 406250#*(1-20*1e-6)
    fs = 1e6
    sig_freq = 2.4e9
    # sig_freq = 2400000030.517578#-52e6/(2**18)
    preamble_len = 16  # TODO!!!!
    skip_preambles = 8  # skip first 8 preambles ## TODO
    total_len = 89
    thresh = None# 0.03
    # file_paths = ['/data/djl/temp/OptimalPkt/fingerprint_data/data0_test_3',]
    base_path = '/data/djl/temp/OptimalPkt/fingerprint_data/'

    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    nsamp = round(n_classes * fs / bw)
    # f_lower, f_upper = -50000, -30000
    f_lower, f_upper = -38000, -34000
    t_lower, t_upper = 0, nsamp
    fguess = (f_lower + f_upper) / 2
    tguess = nsamp / 2
    code_len = 2

    cfo_freq_est = -39705.621
    time_error = 4321.507

    time_r = time_error / tsig
    cfo_r = cfo_freq_est / bw
    # print(time_r + cfo_r, cfo_r - time_r)


    fguess = cfo_freq_est
    tguess = time_error
    f_lower = fguess - 500
    f_upper = fguess + 500
    t_lower = tguess - 50
    t_upper = tguess + 50

    tbeta = (0.00025319588796389843+0.0002531603667156756)/2
    fbeta = (-6.283185307178473e-06-6.302329387166181e-06)/2
    # fguess = -39000
    # tguess = 4320

    gen_refchirp_deadzone = 0
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)

    fft_upsamp = 1024
    detect_range_pkts = 1
    detect_to_max = nsamp * 2
    fft_n = int(fs) #nsamp * fft_upsamp
    plan = fft.get_fft_plan(cp.zeros(fft_n, dtype=cp.complex64))
    fft_ups = cp.zeros((preamble_len + detect_range_pkts, fft_n), dtype=cp.float32)
    fft_downs = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.float32)
    fft_ups_x = cp.zeros((preamble_len + detect_range_pkts, fft_n), dtype=cp.complex64)
    fft_downs_x = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.complex64)
    # fft_ups = cp.zeros( fft_n, dtype=cp.float32)
    # fft_downs = cp.zeros( fft_n, dtype=cp.float32)


if use_gpu:
    cp.cuda.Device(0).use()
Config = Config()
file_paths = [os.path.join(Config.base_path, x) for x in os.listdir(Config.base_path) if 'data0' in x]
Config.file_paths = sorted(file_paths, key=lambda x: int(x.split('_')[-1])) # TODO!



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

def objective(params, pktdata2a):
    cfofreq, time_error = converter_up(*params)
    return objective_core(cfofreq, time_error, pktdata2a)



def objective_linear(cfofreq, time_error, pktdata2a):
    detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error), deadzone=Config.gen_refchirp_deadzone,
                               calctime=0)
    detect_symb_concat = cp.concatenate(detect_symb, axis=0)
    res = cp.zeros(len(detect_symb), dtype=cp.float32)  # complex64)
    ddx = 0  # TODO
    tint = math.ceil(time_error)
    # ress2 = cp.conj(pktdata2a[tint:tint + len(detect_symb_concat)]) * detect_symb_concat
    ress2 = -cp.unwrap(cp.angle(pktdata2a[tint:tint + len(detect_symb_concat)])) + cp.unwrap(cp.angle(detect_symb_concat))
    # plt.plot(cp.unwrap(cp.angle(pktdata2a[tint:tint + len(detect_symb_concat)])).get())
    # plt.plot(cp.unwrap(cp.angle(detect_symb_concat)).get())
    # fig = px.scatter(y=ress2[:150000].get())
    # fig.show()


    phasediff = ress2#cp.unwrap(cp.angle(cp.array(ress2)))
    est_dfreqs = []
    lflag = False
    if lflag:
        fig = px.scatter(y=phasediff[:Config.preamble_len * Config.nsamp].get())
        fig.update_traces(marker=dict(size=2))
    for fit_symbidx in range(0, Config.preamble_len):
        x_values = np.arange(Config.nsamp * fit_symbidx + 50, Config.nsamp * (fit_symbidx + 1) - 50)
        y_values = phasediff[x_values].get()
        coefficients = np.polyfit(x_values, y_values, 1)
        est_dfreq = coefficients[0] * Config.fs / 2 / np.pi
        est_dfreqs.append(est_dfreq)
        # print(f"fitted curve {est_dfreq=:.2f} Hz")
        if lflag: fig.add_trace(go.Scatter(x=x_values, y=np.poly1d(coefficients)(x_values), mode="lines"))
    if lflag:
        fig.update_layout(title = f"{cfofreq:.2f} Hz {time_error:.2f} sps")
        fig.show()

    est_ups = []
    for fit_symbidx in range(Config.sfdpos, Config.sfdpos + 2):
        x_values = np.arange(Config.nsamp * fit_symbidx + 50, Config.nsamp * (fit_symbidx + 1) - 50)
        y_values = phasediff[x_values].get()
        coefficients = np.polyfit(x_values, y_values, 1)
        est_ufreq = coefficients[0] * Config.fs / 2 / np.pi
        est_ups.append(est_ufreq)
        # print(f"fitted curve {est_ufreq=:.2f} Hz")
    ret_ufreq = np.mean(est_ups)
    # fig = px.scatter(y=est_dfreqs)
    ret_dfreq = np.mean(est_dfreqs[Config.skip_preambles:])
    # fig.add_hline(y=ret_dfreq)
    # fig.show()

    beta = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs
    ret_freq = (ret_ufreq + ret_dfreq)/2
    ret_tdiff = (ret_ufreq - ret_dfreq)/2 / beta

    return ret_freq, ret_tdiff
        # objective_core(cfofreq + est_dfreq / 10, time_error, pktdata2a, True, calctime-1)


def objective_core(cfofreq, time_error, pktdata2a, drawflag = False):
    # print('input', cfofreq, time_error, 0)
    if time_error < 0:# or time_error > Config.detect_to_max: # TODO!!!
        # print('ret', cfofreq, time_error, 0)
        return 0
    if abs(cfofreq + 20000) > 20000: return 0 # TODO !!!
    assert pktdata2a.ndim == 1
    assert cp.mean(cp.abs(pktdata2a)).ndim == 0
    # pktdata2a_roll = cp.roll(pktdata2a / cp.mean(cp.abs(pktdata2a)), -math.ceil(time_error))
    detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error), deadzone=Config.gen_refchirp_deadzone, calctime=0)
    detect_symb_concat = cp.concatenate(detect_symb, axis=0)
    if not drawflag:
        res = cp.abs(cp.conj(pktdata2a[math.ceil(time_error):math.ceil(time_error)+len(detect_symb_concat)]).dot(detect_symb_concat)) / len(detect_symb_concat)
        return -res.get()

    res = cp.zeros(len(detect_symb), dtype=cp.float32)#complex64)
    ddx = 0  # TODO
    ress2 = cp.conj(pktdata2a[math.ceil(time_error):math.ceil(time_error) + len(detect_symb_concat)]) * detect_symb_concat

    res2 = []
    for sidx, ssymb in enumerate(detect_symb):
        # a1 = 0.23
        # a2 = -0.13
        # f1 = -a1 * Config.fs / 2 / np.pi
        # f2 = -a2 * Config.fs / 2 / np.pi
        # sigt = 2 ** Config.sf / Config.bw * Config.fs

        # fixchirp = gen_upchirp(0, sigt * 2, f1 * 2, (f2 - f1) / sigt) # not fix length TODO
        symbdata = pktdata2a[ddx + math.ceil(time_error): math.ceil(time_error) + ddx + len(ssymb)] #* fixchirp[:len(ssymb)]
        ress = cp.conj(ssymb).dot(symbdata)
        ddx += len(ssymb)
        # res[sidx] = ress / len(ssymb) # !!! abs: not all add up TODO!!
        res[sidx] = cp.abs(ress) / len(ssymb) # !!! abs: not all add up
        res2.append(cp.angle(ress).get())
    # print(res)
    res2 = np.array(res2)

    # beta = Config.bw / ( 2 ** Config.sf / Config.bw )
    ret =  - tocpu(cp.abs(cp.sum(res)) / len(res-2)) # two zero codes
    # print('ret', cfofreq, time_error, ret)
    if drawflag :#ret<-0.08:
        plt.plot(res2)
        plt.title(f"nounwrap {cfofreq:.2f} Hz {time_error:.2f} sps")
        plt.show()

        plt.plot(np.unwrap(res2))

        x_values = np.arange(Config.skip_preambles, Config.preamble_len)
        y_values = np.unwrap(res2[x_values])
        coefficients = np.polyfit(x_values, y_values, 1)
        plt.plot(x_values, np.poly1d(coefficients)(x_values))
        coef = coefficients[0]
        coef_estcfo = coefficients[0] / Config.bw / (2 ** Config.sf / Config.bw) * Config.sig_freq
        plt.title(f"{cfofreq:.2f} Hz {time_error:.2f} sps {coef:.5e} {coef_estcfo:.2e}")
        logger.info(f"{cfofreq:.2f} Hz {time_error:.2f} sps {coef=} {coef_estcfo=}")
        plt.show()
    if drawflag:# and calctime > 0:


        cumulative_sums = cp.cumsum(cp.array(ress2))
        result_gpu = cp.abs(cumulative_sums)
        result_cpu = result_gpu.get()
        plt.plot(result_cpu)
        for i in range(Config.preamble_len): plt.axvline(Config.nsamp * i)
        for i in range(Config.sfdpos, Config.sfdpos + 3): plt.axvline(Config.nsamp * i)
        plt.title(f"{cfofreq:.2f} Hz {time_error:.2f} sps {ret=}")
        plt.show()
        # phasediff = cp.diff(cp.angle(cp.array(ress2)))
        # fig = px.scatter(y=phasediff[:Config.nsamp*2].get())
        # fig.update_traces(marker=dict(size=2))
        # fig.add_vline(x=Config.nsamp, line=dict(color="black", dash="dash"))
        # fig.show()

    if False:

        phasediff = cp.unwrap(cp.angle(cp.array(ress2)))
        x_values = np.arange(Config.nsamp * (Config.skip_preambles - 2), Config.nsamp * (Config.skip_preambles + 2))
        fig = px.scatter(x=x_values, y=phasediff[x_values].get())
        fig.update_traces(marker=dict(size=2))
        fig.add_vline(x=Config.nsamp, line=dict(color="black", dash="dash"))

        x_values = np.arange(Config.nsamp * Config.skip_preambles + 50, Config.nsamp * (Config.skip_preambles + 1) - 50)
        y_values = phasediff[x_values].get()
        coefficients = np.polyfit(x_values, y_values, 1)
        est_dfreq = coefficients[0] * Config.fs
        print(f"fitted curve {est_dfreq=:.2f} Hz")
        fig.add_trace(go.Scatter(x=x_values, y=np.poly1d(coefficients)(x_values), mode="lines"))
        fig.update_layout(title = f"{cfofreq:.2f} Hz {time_error:.2f} sps {ret=:.3f} {calctime=} {est_dfreq=:.2f} Hz")
        fig.show()
        # objective_core(cfofreq + est_dfreq / 10, time_error, pktdata2a, True, calctime-1)


        # sys.exit(0)

        # plt.plot(cp.abs(cp.array(ress2)).get())
        # plt.show()
        # print(ress, - tocpu(cp.abs(cp.sum(res))))
        # print(cp.mean(cp.abs(pktdata2a[:ddx])))  # Negative because we use a minimizer
        # TODO qian zhui he plot
        # TODO remove **2 because res[sidx] is sum not sumofsquare
        # TODO phase consistency?
    return ret



def gen_refchirp(cfofreq, tstart, deadzone=0, calctime=0):
    detect_symb = []
    bw = Config.bw * (1 + cfofreq / Config.sig_freq)
    sigt = 2 ** Config.sf / bw * Config.fs #* (1 - cfofreq / Config.sig_freq)
    beta = bw / sigt
    sigt *= (1 + 1e-6 * calctime)# (1 + cfofreq / Config.sig_freq)
    # print(Config.bw / Config.tsig * (1 + cfofreq / Config.sig_freq) / (1 - cfofreq / Config.sig_freq))
    # print(Config.bw / Config.tsig * (1 + cfofreq / Config.sig_freq) )
    # print(Config.bw / Config.tsig )
    for tid in range(Config.preamble_len):
        upchirp = gen_upchirp(tstart + sigt * tid, sigt, -bw  / 2 + cfofreq, beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    for tid in range(Config.preamble_len, Config.sfdpos):
        detect_symb.append(cp.zeros(math.ceil(tstart + sigt * (tid + 1)) - math.ceil(tstart + sigt * tid), dtype=cp.complex64))
    for tid in range(Config.sfdpos, Config.sfdend):
        upchirp = gen_upchirp(tstart + sigt * tid, sigt if tid != Config.sfdend - 1 else sigt / 4,
                              bw / 2 + cfofreq, - beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    return detect_symb




def read_large_file(file_path_in):
    with open(file_path_in, 'rb') as file:
        # t = 1.45e6
        while True:
            try:
                rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp)
                # t-=len(rawdata)
            except EOFError:
                logger.warning(f"file complete with EOF {file_path_in=}")
                break
            if len(rawdata) < Config.nsamp:
                logger.warning(f"file complete{file_path_in=}, {len(rawdata)=}")
                break
            # if t<0:
            #     plt.scatter(x=np.arange(Config.nsamp - 1),
            #                 y=cp.diff(cp.unwrap(cp.angle(rawdata[:Config.nsamp]))).get(), s=0.2)
            #     plt.show()
            yield rawdata



def read_pkt(file_path_in1, file_path_in2, threshold, min_length=20):
    current_sequence1 = []
    current_sequence2 = []

    read_idx = -1
    for rawdata1, rawdata2 in itertools.zip_longest(read_large_file(file_path_in1), read_large_file(file_path_in2)):
        if rawdata1 is None or rawdata2 is None:
            break  # Both files are done
        read_idx += 1

        number1 = cp.max(cp.abs(rawdata1)) if rawdata1 is not None else 0

        # Check for threshold in both files
        if number1 > threshold:
            current_sequence1.append(rawdata1)
            current_sequence2.append(rawdata2)
        else:
            if len(current_sequence1) > min_length:
                yield read_idx, cp.concatenate(current_sequence1), cp.concatenate(current_sequence2)
            current_sequence1 = []
            current_sequence2 = []

    # Yield any remaining sequences after the loop
    if len(current_sequence1) > min_length:
        yield read_idx, cp.concatenate(current_sequence1), cp.concatenate(current_sequence2)


def add_freq(pktdata_in, est_cfo_freq):
    cfosymb = cp.exp(2j * cp.pi * est_cfo_freq * cp.linspace(0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
    cfosymb = cfosymb.astype(cp.complex64)
    pktdata2a = pktdata_in * cfosymb
    return pktdata2a

def coarse_work_fast(pktdata_in, fstart, tstart , retpflag = False, linfit = False, sigD = False):

    # tstart = round(tstart) # !!!!! TODO tstart rounded !!!!!

    # plot angle of input
    # plt.plot(cp.unwrap(cp.angle(pktdata_in)).get()[round(tstart):round(tstart)+Config.nsamp*20])
    # plt.axvline(Config.nsamp)
    # plt.axvline(Config.nsamp*2)
    # plt.show()

    est_to_s = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
    cfoppm = fstart / Config.sig_freq
    t1 = 2 ** Config.sf / Config.bw * (1 - cfoppm)
    upchirp = mychirp(est_to_s, f0=-Config.bw / 2, f1=Config.bw / 2, t1=t1)
    downchirp = mychirp(est_to_s, f0=Config.bw / 2, f1=-Config.bw / 2, t1=t1)

    fft_sig_n = Config.bw#  round(Config.bw / Config.fs * Config.fft_n) # 4096 fft_n=nsamp*fft_upsamp, nsamp=t*fs=2**sf/bw*fs, fft_sig_n=2**sf * fft_upsamp
    fups = [] # max upchirp positions
    fret = [] # return phase of two peaks

    # upchirp dechirp
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    fdiff = []
    for pidx in range(Config.preamble_len + Config.detect_range_pkts): # assume chirp start at one in [0, Config.detect_range_pkts) possible windows
        assert Config.nsamp * pidx + tstart >= 0
        start_pos_all = nsamp_small * pidx + tstart
        start_pos = round(start_pos_all)
        start_pos_d = start_pos_all - start_pos
        fdiff.append(start_pos_d)
        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
        sig2 = sig1 * downchirp
        if not linfit:
            # use input cfo for sfo
            sig2 = add_freq(sig2, - fstart/Config.sig_freq * Config.bw * pidx + start_pos_d / nsamp_small * Config.bw / Config.fs * Config.fft_n)
            # pass
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        data = cp.abs(data0) + cp.abs(cp.roll(data0, -fft_sig_n))
        Config.fft_ups[pidx] = data
        Config.fft_ups_x[pidx] = data0
        amax = cp.argmax(cp.abs(data))
        fups.append(amax.get())

    # downchirp
    for pidx in range(Config.sfdpos, Config.sfdpos + 2 + Config.detect_range_pkts):

        assert Config.nsamp * pidx + tstart >= 0
        start_pos_all = nsamp_small * pidx + tstart
        start_pos = round(start_pos_all)
        start_pos_d = start_pos_all - start_pos
        fdiff.append(start_pos_d)
        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
        sig2 = sig1 * upchirp
        if not linfit:
            # use input cfo for sfo
            sig2 = add_freq(sig2,
                             fstart / Config.sig_freq * Config.bw * pidx + start_pos_d / nsamp_small * Config.bw / Config.fs * Config.fft_n)
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        data = cp.abs(data0) + cp.abs(cp.roll(data0, -fft_sig_n))
        Config.fft_downs[pidx - Config.sfdpos] = data

    if retpflag:
        # draw_fit(0, pktdata_in, 0, tstart)
        return fret


    # fit the up chirps with linear, intersect with downchirp
    detect_vals = np.zeros((Config.detect_range_pkts, 3))
    for detect_pkt in range(Config.detect_range_pkts): # try all possible starting windows, signal start at detect_pkt th window


        # linear fit fups
        fdiff = np.array(fdiff)
        if linfit:
            y_values = fups[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt] + fdiff[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt] / nsamp_small * Config.bw / Config.fs * Config.fft_n
            y_values = [(x + fft_sig_n//2) % fft_sig_n - fft_sig_n//2 for x in y_values]  # move into [-fft_sig_n//2, fft_sig_n//2] range
            # print(f"{fft_sig_n=}")
            # y_values = y_values % Config.bw
            x_values = np.arange(len(y_values)) + Config.skip_preambles + detect_pkt
            coefficients = np.polyfit(x_values, y_values, 1)

            # plot fitted result
            # plt.scatter(x_values, y_values)
            # plt.plot(x_values, np.poly1d(coefficients)(x_values))
            # plt.title("linear")
            # plt.show()
            # print('c',coefficients[0]/Config.bw*Config.fs)

            polynomial = np.poly1d(coefficients)
        else:
            y_value_debug = cp.argmax(cp.sum(cp.abs(Config.fft_ups[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt, :fft_sig_n]), axis=0)).get()
            # for i in range(Config.skip_preambles + detect_pkt, Config.preamble_len + detect_pkt):
            #     plt.plot(Config.fft_ups[i].get())
            # plt.show()
            Config.fft_ups[:, fft_sig_n:] = 0 # !!!
            fig = go.Figure()
            for i in range(Config.skip_preambles + detect_pkt, Config.preamble_len + detect_pkt):
                fig.add_trace(go.Scatter(y=Config.fft_ups[i, y_value_debug - 500: y_value_debug + 500].get(), mode="lines"))

            fig.show()


            y_value = cp.argmax(cp.sum(cp.abs(Config.fft_ups[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt]), axis=0)).get()
            k = fstart/Config.sig_freq * Config.bw
            coefficients = (k, y_value - (Config.skip_preambles + detect_pkt) * k)
            polynomial = np.poly1d(coefficients)

            y_values = fups[Config.skip_preambles + detect_pkt: Config.preamble_len + detect_pkt]
            print('y ys', y_value, y_values, fft_sig_n)


            y_values = [(x + fft_sig_n // 2) % fft_sig_n - fft_sig_n // 2 for x in y_values]  # move into [-fft_sig_n//2, fft_sig_n//2] range
            x_values = np.arange(len(y_values)) + Config.skip_preambles + detect_pkt
            plt.scatter(x_values, y_values)
            plt.plot(x_values, polynomial(x_values))
            plt.title("linear")
            plt.show()
            print('c-nolinear',k/Config.bw*Config.fs)

            if sigD:
                def dirichlet_kernel(frequency, f0, N, fs):
                    f0 = f0.item()
                    omega = 2 * cp.pi * (frequency - f0) / fs
                    result = cp.sin(N * omega / 2) / cp.sin(omega / 2)
                    result = result.astype(cp.complex64)
                    result *= cp.exp(-1j * 2 * np.pi / (fs / N) * (frequency - f0))
                    result[omega == 0] = N  # Handle the zero division case
                    return result

                # Define the cost function (MSE)
                def cost_function(params, frequency, fft_result, N, fs):
                    f0 = params
                    dirichlet = dirichlet_kernel(frequency, f0, N, fs)
                    mse = - cp.abs(fft_result.dot(cp.conj(dirichlet))).get().item()
                    return mse

                for i in range(Config.skip_preambles + detect_pkt, Config.preamble_len + detect_pkt):
                    y_value = cp.argmax( cp.abs(Config.fft_ups_x[i])).get() # use individual or same y-value
                    initial_guess = y_value
                    frequency_shifted = cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))
                    result = minimize(cost_function, initial_guess, args=(frequency_shifted, Config.fft_ups_x[i], Config.nsamp, Config.fs))
                    estimated_f0 = result.x[0]


                    # Plot the cost function over a range around y_value
                    f_range = cp.linspace(y_value - 1000, y_value + 1000, 500)
                    cost_values = cp.array([cost_function(f0, frequency_shifted, Config.fft_ups_x[i] , Config.nsamp, Config.fs) for f0 in f_range])

                    # Convert to numpy for plotting
                    estimated_f0 = f_range[cp.argmin(cost_values)].get()
                    # estimated_f0 = y_value # TODO!!!

                    dirichlet = dirichlet_kernel( cp.linspace(0, Config.fs, Config.fft_n, endpoint=False), estimated_f0, Config.nsamp, Config.fs)
                    estimated_phase = cp.angle(cp.sum(Config.fft_ups_x[i] * dirichlet))

                    dirichlet0 = dirichlet_kernel( cp.linspace(0, Config.fs, Config.fft_n, endpoint=False), y_value, Config.nsamp, Config.fs)
                    if True:
                        plt.figure(figsize=(12, 6))
                        plt.plot(f_range.get(), cost_values.get())
                        plt.title("Cost Function Over Frequency Range")
                        plt.axvline(y_value, color='r')
                        plt.axvline(estimated_f0, color='k')
                        plt.xlabel("Frequency (Hz)")
                        plt.ylabel("cost function")
                        plt.grid(True)
                        plt.show()

                    if True:
                        maxn = cp.max(cp.abs(Config.fft_ups_x[i])).get().item()
                        fig = go.Figure()
                        fig.add_trace( go.Scatter(y=np.abs(Config.fft_ups_x[i, y_value - 500: y_value + 500].get()), mode="markers"))
                        fig.add_trace( go.Scatter(y=np.abs(dirichlet[y_value - 500: y_value + 500].get()), mode="markers"))
                        # fig.add_trace( go.Scatter(y=np.abs(Config.fft_ups_x[i ].get()), mode="lines"))
                        # fig.add_trace( go.Scatter(y=np.abs(dirichlet.get() ), mode="lines"))
                        fig.show()


                    # Display the results
                    print(f"{i}th guess: {y_value} Estimated : {estimated_f0} Hz {estimated_phase} radians")
                    # fig = go.Figure()
                    # fig.add_trace(go.Scatter(y=np.unwrap(np.angle(Config.fft_ups_x[i, :].get())), mode="lines"))
                    # fig.add_trace(go.Scatter(y=np.unwrap(np.angle(dirichlet.get() * np.exp(1j * estimated_phase.get()))), mode="lines"))
                    # fig.show()
                    plt.plot(np.unwrap(np.angle(Config.fft_ups_x[i, :].get())) - np.unwrap(np.angle(dirichlet0.get() * np.exp(1j * estimated_phase.get()))))
                    # plt.plot(np.unwrap(np.angle(dirichlet0.get() * np.exp(1j * estimated_phase.get()))))
                    plt.show()




        # the fitted line intersect with the fft_val_down, compute the fft_val_up in the same window with fft_val_down (at fdown_pos)
        # find the best downchirp among all possible downchirp windows
        fdown_pos, fdown = cp.unravel_index(cp.argmax(cp.abs(Config.fft_downs[detect_pkt: detect_pkt + 2])), Config.fft_downs[detect_pkt: detect_pkt + 2].shape)
        fdown_pos = fdown_pos.item() + detect_pkt + Config.sfdpos  # position of best downchirp
        fdown = fdown.item()  # freq of best downchirp

        fft_val_up = (polynomial(fdown_pos) - (Config.fft_n//2)) / fft_sig_n # rate, [-0.5, 0.5) if no cfo and to it should be zero  #!!! because previous +0.5
        fft_val_up = (fft_val_up + 0.5) % 1 - 0.5 # remove all ">0 <0 stuff, just [-0.5, 0.5)

        fft_val_down = (fdown-(Config.fft_n//2)) / fft_sig_n # [0, 1)
        fft_val_down = (fft_val_down + 0.5) % 1 - 0.5 # remove all ">0 <0 stuff, just [-0.5, 0.5)

        f0 = ((fft_val_up + fft_val_down) / 2) % 1
        t0 = (f0 - fft_val_up) % 1

        # try all possible variations (unwrap f0, t0 if their real value exceed [-0.5, 0.5))
        # print(f0, t0)
        deltaf, deltat = np.meshgrid((np.arange(-1, 1.5, 0.5)+f0)*Config.bw, (np.arange(-1, 1.5, 0.5)+t0)*Config.tsig + tstart + detect_pkt*Config.nsamp)
        values = np.zeros_like(deltaf).astype(float)
        for i in range(deltaf.shape[0]):
            for j in range(deltaf.shape[1]):
                values[i][j] = objective_core(deltaf[i,j], deltat[i,j], pktdata_in) # objective_core returns minus power value, so argmin; out of range objective_core=0
        best_idx = np.argmin(values)
        est_cfo_f = deltaf.flat[best_idx]
        est_to_s = deltat.flat[best_idx]
        dvals = -np.min(values)
        detect_vals[detect_pkt] = (dvals, est_cfo_f, est_to_s) # save result

        print('c',coefficients[0], 'd', est_cfo_f/Config.sig_freq * Config.bw, fdiff / nsamp_small * Config.bw / Config.fs * Config.fft_n)

    # find max among all detect windows
    print(f"{detect_vals[:, 0]=}")
    detect_pkt_max = np.argmax(detect_vals[:, 0])
    est_cfo_f, est_to_s = detect_vals[detect_pkt_max, 1], detect_vals[detect_pkt_max, 2]
    # assert detect_pkt_max == 0
    return est_cfo_f, est_to_s


def fix_cfo_to(est_cfo_f, est_to_s, pktdata_in):
    est_cfo_slope = est_cfo_f / Config.sig_freq * Config.bw / Config.tsig * Config.fs
    sig_time = len(pktdata_in) / Config.fs
    # logger.info(f'I03_3: SFO {est_cfo_f=} Hz, {est_cfo_slope=} Hz/s, {sig_time=} s')
    estt = cp.linspace(0, sig_time, len(pktdata_in) + 1)[:-1]
    est_cfo_symbol = mychirp(estt, f0=-est_cfo_f, f1=-est_cfo_f + est_cfo_slope * sig_time, t1=sig_time)
    pktdata_fix = pktdata_in * est_cfo_symbol
    return cp.roll(pktdata_fix, -round(est_to_s))


# read packets from file
if __name__ == "__main__":
    ps = []
    ps2 = []
    psa1 = []
    psa2 = []
    ps3 = []
    fulldata = []

    # Main loop read files
    for file_path in Config.file_paths:
        file_path = "hou2"

        #  read file and count size
        file_path_id = 0# int(file_path.split('_')[-1])

        logger.info(f"FILEPATH { file_path}")
        pkt_cnt = 0
        pktdata = []
        fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
        logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')

        # read max power of first 5000 windows, for envelope detection
        power_eval_len = 5000
        nmaxs = []
        for idx, rawdata in enumerate(read_large_file(file_path)):
            nmaxs.append(cp.max(cp.abs(rawdata)))
            if idx == power_eval_len - 1: break
        nmaxs = cp.array(nmaxs).get()

        # clustering
        data = nmaxs.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(data)
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        sorted_indices = np.argsort(means)
        mean1, mean2 = means[sorted_indices]
        covariance1, covariance2 = covariances[sorted_indices]
        weight1, weight2 = weights[sorted_indices]

        # threshold to divide the noise power from signal power
        thresh = (mean1 * covariance2 + mean2 * covariance1) / (covariance1 + covariance2)

        # if threshold may not work set this to True
        # plot the power map
        if False:
            counts, bins = cp.histogram(nmaxs, bins=100)
            # logger.debug(f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")
            logger.debug(f"cluster: {kmeans.cluster_centers_[0]} {kmeans.cluster_centers_[1]} {thresh=}")
            threshpos = np.searchsorted(tocpu(bins), thresh).item()
            logger.debug(f"lower: {cp_str(counts[:threshpos])}")
            logger.debug(f"higher: {cp_str(counts[threshpos:])}")
            fig = px.line(nmaxs.get())
            fig.add_hline(y=thresh)
            fig.update_layout(
                title=f"{file_path} pow {len(nmaxs)}",
                legend=dict(x=0.1, y=1.1))
            fig.show()

        # loop for demodulating all decoded packets: iterate over pkts with energy>thresh and length>min_length
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, file_path.replace("data0", "data1"), thresh, min_length=30)):

            # read data: read_idx is the index of packet end window in the file
            read_idx, data1, data2 = pkt_data
            # (Optional) skip the first pkt because it may be half a pkt. read_idx == len(data1) means this pkt start from start of file
            if read_idx == len(data1) // Config.nsamp: continue

            # normalization
            data1 /= cp.mean(cp.abs(data1))
            data2 /= cp.mean(cp.abs(data1))
            # data1 = cp.concatenate((cp.zeros(Config.nsamp//2, dtype=data1.dtype), data1))
            # data2 = cp.concatenate((cp.zeros(Config.nsamp//2, dtype=data2.dtype), data2))

            logger.info(f"Prework {pkt_idx=} {len(data1)=}")
            est_cfo_f = 0
            est_to_s = 0
            trytimes = 2
            vals = np.zeros((trytimes, 3))
            # iterate trytimes times to detect, each time based on estimations of the last time
            for i in range(trytimes):

                    # main detection function with up-down
                    f, t = coarse_work_fast(data1, est_cfo_f, est_to_s, False, False, i >= 1)

                    # plot error
                    if t < 0:
                        # plot unwrapped phases of the pkt
                        logger.error(f"ERROR in {est_cfo_f=} {est_to_s=} out {f=} {t=} {file_path=} {pkt_idx=}")
                        plt.plot(cp.unwrap(cp.angle(data1)).get()[:Config.nsamp*20])
                        plt.show()

                        # plt the powers in nmaxs to see if there is a pkt detection error (variation in recv power)
                        xrange1 = np.arange(read_idx - len(data1) // Config.nsamp, read_idx)
                        filtered_x = [x for x in range(len(nmaxs)) if x not in xrange1]
                        filtered_y = [nmaxs[x] for x in filtered_x]
                        fig = px.scatter(x=filtered_x, y=filtered_y, title=file_path)
                        fig.add_hline(y=thresh, line=dict(dash='dash'))
                        fig.add_trace(go.Scatter(x=xrange1, y=[nmaxs[x] for x in xrange1], mode="markers",line=dict(color='red'),marker=dict(size=3),
                                                 name='Data Range'))
                        fig.update_traces(marker=dict(size=3))
                        fig.show()

                        # use an arbitarily set start iteration point, see if error can be recovered !!!
                        est_cfo_f = -40000
                        est_to_s = random.randint(0, Config.nsamp - 1)
                        break

                    # renew the result for next iteration
                    est_cfo_f = f
                    est_to_s = t
                    objval=objective_core(est_cfo_f, est_to_s, data1)
                    logger.info(f"try{i} {est_cfo_f=} {est_to_s=} obj={objval}")
                    vals[i] = (objval, est_cfo_f, est_to_s)

                    # the best result among trytimes
                    _, est_cfo_f, est_to_s = vals[np.argmin(vals[:, 0])]

                    # fine grained optimization, search around the up-down result towards the local minima
                    Config.fguess = est_cfo_f
                    Config.tguess = est_to_s
                    Config.f_lower = Config.fguess - 500
                    Config.f_upper = Config.fguess + 500
                    Config.t_lower = Config.tguess - 50
                    Config.t_upper = Config.tguess + 50
                    bestobj = objective_core(Config.fguess, Config.tguess, data1)

                    tryidx = 0

                    # alternative method for optimization (doesn't always get good results, don't know why)

                    cfo_freq_est, time_error = est_cfo_f, est_to_s
                    logger.info(
                        f"updown parameters:{cfo_freq_est=} {time_error=} obj={objective_core(cfo_freq_est, time_error, data1, True)}")
                    # for i in range(3):
                    #     linear_dfreq,linear_dtime = objective_linear(cfo_freq_est, time_error, data1)
                    #     logger.info(f"{linear_dfreq=} {linear_dtime=}")
                    #     cfo_freq_est -= linear_dfreq
                    #     time_error -= linear_dtime
                    #     objective_core(cfo_freq_est, time_error, data1, True)
                    # cfo_freq_est -= 2 * linear_dfreq
                    # objective_core(cfo_freq_est, time_error, data1, True)

                    # sys.exit(0)
                    # xv = np.linspace(-100, 100, 1000)
                    # yv = np.array([objective_core(cfo_freq_est, time_error, data1, False, i) for i in xv])
                    # plt.plot(xv, yv)
                    # plt.title("xv yv")
                    # xva = xv[np.argmin(yv)]
                    # plt.axvline(xva)
                    # plt.show()
                    # objective_core(cfo_freq_est, time_error, data1, True, xva)
                    # sys.exit(0)

                    # cfo_freq_est = 4.60472836e+02
                    # logger.info(
                    #     f"updown parameters:\n{cfo_freq_est=}\n{time_error=} obj={objective_core(cfo_freq_est, time_error, data1, True)}")

                    if False:
                        cfo_freq_est, time_error = est_cfo_f, est_to_s
                        tryidx += 1
                        start_t = est_to_s#random.uniform(Config.t_lower, Config.t_upper)
                        start_f = est_cfo_f#random.uniform(Config.f_lower, Config.f_upper)
                        result = opt.minimize(objective, converter_down(start_f, start_t), args=(data1,),
                                              bounds=[
                                                  (converter_down(Config.f_lower, 0)[0], converter_down(Config.f_upper, 0)[0]),
                                                  (converter_down(0, Config.t_lower)[1], converter_down(0, Config.t_upper)[1])],
                                              method='L-BFGS-B',
                                              options={'gtol': 1e-12, 'disp': False}
                                              )

                        if result.fun < bestobj:
                            cfo_freq_est, time_error = converter_up(*result.x)
                            logger.debug(
                                f"{tryidx=: 6d} cfo_freq_est = {cfo_freq_est:.3f}, time_error = {time_error:.3f} {result.fun=:.5f}")
                            # Config.fguess, Config.tguess = result.x
                            bestobj = result.fun
                            if tryidx > 100: draw_fit(pktidx, data1, cfo_freq_est, time_error)
                        if tryidx == 100: draw_fit(pktidx, data1, cfo_freq_est, time_error)
                        logger.info(f"Optimized parameters:\n{cfo_freq_est=}\n{time_error=} obj={objective_core(cfo_freq_est, time_error, data1, True)}")
                        # draw_fit(0, data1, cfo_freq_est, time_error)
                        est_cfo_f, est_to_s = cfo_freq_est, time_error

                    # skip fine_grained optimization

                    # compute angles
                    data_angles = []
                    for pidx in range(Config.total_len):
                        sig1 = data1[Config.nsamp * pidx + est_to_s: Config.nsamp * (pidx + 1) + est_to_s]
                        sig2 = data2[Config.nsamp * pidx + est_to_s: Config.nsamp * (pidx + 1) + est_to_s]
                        data_angles.append((sig1.dot(sig2.conj())).item())
                    # save data for output line
                    fulldata.append([file_path_id, est_cfo_f, est_to_s, *(np.angle(np.array(data_angles))), *(np.abs(np.array(data_angles))) ])
                    # save data for plotting
                    ps.extend(data_angles)
                    psa1.append(len(ps))
                    ps2.append(est_cfo_f)
                    ps3.append(est_to_s)
            sys.exit(0)
        # the length of each pkt (for plotting)
        psa1 = psa1[:-1]
        psa2.append(len(ps))

        # save info of all the file to csv (done once each packet, overwrite old)
        if True:
            header = ["fileID", "CFO", "Time offset"]
            header.extend([f"Angle{x}" for x in range(Config.total_len)])
            header.extend([f"Abs{x}" for x in range(Config.total_len)])
            csv_file_path = 'output_with_header.csv'
            with open(csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(header)  # Write the header
                for row in fulldata:
                    csvwriter.writerow(row)


        # Plot for the information of all the file (done once each packet, overwrite old)
        if False:
            fig1 = go.Figure()

            fig1.add_trace(go.Scatter(
                x=np.arange(len(np.angle(np.array(ps)))),
                y=np.angle(np.array(ps)),
                mode='lines',
                name='Angle of ps'
            ))

            # Add vertical lines for psa1 (blue dashed) and psa2 (black solid)
            for i in psa1:
                fig1.add_vline(x=i, line=dict(color="blue", dash="dash"))

            for i in psa2:
                fig1.add_vline(x=i, line=dict(color="black", dash="solid"))

            # Save the figure
            fig1.write_html("angles.html")

            # Plot for the magnitude of ps
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=np.arange(len(np.abs(np.array(ps)))),
                y=np.abs(np.array(ps)),
                mode='lines',
                name='Magnitude of ps'
            ))

            # Add vertical lines for psa1 (blue dashed) and psa2 (black solid)
            for i in psa1:
                fig2.add_vline(x=i, line=dict(color="blue", dash="dash"))

            for i in psa2:
                fig2.add_vline(x=i, line=dict(color="black", dash="solid"))

            # Save the figure
            fig2.write_html("signal amplitudes.html")

            # Plot for the magnitude of ps2
            fig3 = go.Figure()

            fig3.add_trace(go.Scatter(
                x=np.arange(len(np.abs(np.array(ps2)))),
                y=np.abs(np.array(ps2)),
                mode='lines',
                name='Magnitude of ps2'
            ))

            # Save the figure
            fig3.write_html("estimated cfo.html")

            # Plot for the magnitude of ps3 as a scatter plot
            fig4 = go.Figure()

            fig4.add_trace(go.Scatter(
                x=np.arange(len(np.abs(np.array(ps3)))),
                y=np.abs(np.array(ps3)),
                mode='markers',
                name='Magnitude of ps3'
            ))

            # Save the figure
            fig4.write_html("estimated time offsets.html")

