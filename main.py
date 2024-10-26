import argparse
import logging
import os
import random
import sys
import time
import pickle
import cmath
import math
# import matplotlib.pyplot as plt, mpld3
import plotly.express as px
import plotly.graph_objects as go
# import pandas as pd
# Enable fallback mode
# from cupyx.fallback_mode import numpy as np
import numpy as np
import scipy.optimize as opt
# import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from tqdm import tqdm
# import scipy

logger = logging.getLogger('my_logger')
level = logging.DEBUG
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(level)  # Set the console handler level
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(level)  # Set the file handler level
formatter = logging.Formatter('%(message)s')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', default=False, help='Use cpu instead of gpu (numpy instead of cupy)')
parser.add_argument('--searchphase', action='store_true', default=False)
parser.add_argument('--searchphase_step', type=int, default=1000)
parser.add_argument('--searchfromzero', action='store_true', default=False)
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
    renew_switch = False
    s1 = 0
    s2 = 1
    s3 = 1
    s4 = 1
    s5 = 1
    s6 = 1
    s7 = 1
    s8 = 1
    sflag_end = None

    if parse_opts.noplot:
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 0
        s5 = 0
        s6 = 0
        s7 = 0
        s8 = 0

    sf = 7
    bw = 406250
    fs = 1e6
    sig_freq = 2.4e9
    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)
    # file_paths = ['/data/djl/datasets/Dataset_50Nodes/sf7-470-new-70.bin']
    # file_paths = ['/data/djl/datasets/sf7-470-pre-2.bin']
    # file_paths = ['/data/djl/datasets/sf7-470-pre-2.bin']

    base_dir = '/data/djl/datasets/Dataset_50Nodes'
    file_paths = ['/data/djl/temp/OptimalPkt/data1_test']
    # for file_name in os.listdir(base_dir):
    #     if file_name.startswith('sf7') and file_name.endswith('.bin'):
    #         file_paths.append(os.path.join(base_dir, file_name))

    nsamp = round(n_classes * fs / bw)

    preamble_len = 60  # TODO
    code_len = 2
    # codes = [50, 101]  # TODO set codes
    fft_upsamp = 1024
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3
    debug = True
    breakflag = True

    t = cp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    # if not gpu: t = t.get()
    # logger.debug(type(t))
    upchirp = mychirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw)
    downchirp = mychirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw)
    if use_gpu:
        plans = {1: fft.get_fft_plan(cp.zeros(nsamp * 1, dtype=cp.complex64)), fft_upsamp: fft.get_fft_plan(cp.zeros(nsamp * fft_upsamp, dtype=cp.complex64))}
    else:
        plans = {1: None, fft_upsamp: None}

    dataE1 = cp.zeros((n_classes, nsamp), dtype=cp.complex64)
    dataE2 = cp.zeros((n_classes, nsamp), dtype=cp.complex64)
    for symbol_index in range(n_classes):
        time_shift = int(symbol_index / n_classes * nsamp)
        time_split = nsamp - time_shift
        dataE1[symbol_index][:time_split] = downchirp[time_shift:]
        if symbol_index != 0: dataE2[symbol_index][time_split:] = downchirp[:time_shift]
    dataE1 = cp.array(dataE1)
    dataE2 = cp.array(dataE2)


if use_gpu:
    cp.cuda.Device(0).use()
opts = Config()
Config = Config()


def myfft(chirp_data, n, plan):
    if use_gpu:
        return fft.fft(chirp_data, n=n, plan=plan)
    else:
        return fft.fft(chirp_data, n=n)


def dechirp_phase(ndata, refchirp, upsamp=None):
    if len(ndata.shape) == 1:
        ndata = ndata.reshape(1, -1)
    if not upsamp:
        upsamp = Config.fft_upsamp
    # upsamp = Config.fft_upsamp #!!!
    chirp_data = ndata * refchirp
    ans = cp.zeros(ndata.shape[0], dtype=float)
    varphi = cp.zeros(ndata.shape[0], dtype=cp.complex64)
    for idx in range(ndata.shape[0]):
        fft_raw = myfft(chirp_data[idx], n=Config.nsamp * upsamp, plan=Config.plans[upsamp])
        target_nfft = Config.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(float) / upsamp
        varphi[idx] = cut1[tocpu(cp.argmax(dat))]
        varphi[idx] /= abs(varphi[idx])
    return ans, varphi


# noinspection SpellCheckingInspection
def dechirp(ndata, refchirp, upsamp=None):
    if len(ndata.shape) == 1:
        ndata = ndata.reshape(1, -1)
    if not upsamp:
        upsamp = Config.fft_upsamp
    # upsamp = Config.fft_upsamp #!!!
    chirp_data = ndata * refchirp
    ans = cp.zeros(ndata.shape[0], dtype=float)
    power = cp.zeros(ndata.shape[0], dtype=float)
    for idx in range(ndata.shape[0]):
        fft_raw = myfft(chirp_data[idx], n=Config.nsamp * upsamp, plan=Config.plans[upsamp])
        target_nfft = Config.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(float) / upsamp
        power[idx] = cp.max(dat)  # logger.debug(cp.argmax(dat), upsamp, ans[idx])
    return ans, power


def add_freq(pktdata_in, est_cfo_freq):
    cfosymb = cp.exp(2j * cp.pi * est_cfo_freq * cp.linspace(0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
    pktdata2a = pktdata_in * cfosymb
    return pktdata2a


def coarse_work(pktdata_in):
    argmax_est_time_shift_samples = 0
    argmax_est_cfo_samples = 0
    argmax_val = 0
    fft_n = Config.nsamp * Config.fft_upsamp
    # integer detection
    for est_time_shift_samples in tqdm(range(Config.nsamp * 2), disable=not Config.debug):

        fft_raw = cp.zeros(fft_n, dtype=float)
        for preamble_idx in range(Config.preamble_len):
            sig1_pos = est_time_shift_samples + Config.nsamp * preamble_idx
            sig1 = pktdata_in[sig1_pos: sig1_pos + Config.nsamp] * Config.downchirp
            fft_raw_1 = myfft(sig1, n=fft_n, plan=Config.plans[Config.fft_upsamp])
            fft_raw += cp.abs(fft_raw_1) ** 2
        for sfd_idx in range(Config.sfdpos, Config.sfdpos + 2):
            sig2_pos = est_time_shift_samples + Config.nsamp * sfd_idx
            sig2 = pktdata_in[sig2_pos: sig2_pos + Config.nsamp] * Config.upchirp
            fft_raw_2 = myfft(sig2, n=fft_n, plan=Config.plans[Config.fft_upsamp])
            fft_raw += cp.abs(fft_raw_2) ** 2
        max_val = cp.max(fft_raw)
        if max_val > argmax_val:
            argmax_val = max_val
            argmax_est_time_shift_samples = est_time_shift_samples
            argmax_est_cfo_samples = cp.argmax(fft_raw)
    if argmax_est_cfo_samples > fft_n / 2:
        argmax_est_cfo_samples -= fft_n
    est_cfo_freq = argmax_est_cfo_samples * (Config.fs / fft_n)
    est_to_s = argmax_est_time_shift_samples / Config.fs
    logger.info(f'coarse work: {argmax_est_time_shift_samples=}, {argmax_est_cfo_samples=}, {fft_n=}, {est_cfo_freq=} Hz, {est_to_s=} s)')
    return est_cfo_freq, argmax_est_time_shift_samples


def work(pkt_totcnt, pktdata_in):
    # pktdata_in /= cp.mean(cp.abs(pktdata_in))
    # time_error, cfo_freq_est = fine_work_new(pktdata_in)

    est_cfo_freq, argmax_est_time_shift_samples = coarse_work(pktdata_in)
    pktdata2a = add_freq(pktdata_in, - est_cfo_freq)
    pktdata2a = cp.roll(pktdata2a, -argmax_est_time_shift_samples)
    # second detection
    # ====
    est_to_dec, est_to_int, pktdata3, re_cfo_0, re_cfo_freq, detect = fine_work(pktdata2a)
    logger.debug(f"coarse work {re_cfo_0=}, {est_to_int=}, {est_to_dec=}")
    all_cfo_freq = re_cfo_freq + est_cfo_freq

    detect, upsamp = test_preamble(est_to_dec, pktdata3)

    pktdata4 = pktdata3[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]

    ans1n, ndatas = decode_payload(detect, est_to_dec, pktdata4, pkt_totcnt)

    logger.debug(f'decoded data {cp_str(ans1n)}')
    # sfo correction
    est_cfo_slope = all_cfo_freq / Config.sig_freq * Config.bw * Config.bw / Config.n_classes

    sig_time = len(pktdata3) / Config.fs
    logger.debug(f'{all_cfo_freq=} Hz, {est_cfo_slope=} Hz/s, {sig_time=} samples')
    t = cp.linspace(0, sig_time, len(pktdata3) + 1)[:-1]
    est_cfo_symbol = mychirp(t, f0=0, f1=- est_cfo_slope * sig_time, t1=sig_time)
    pktdata5 = pktdata3 * est_cfo_symbol
    detect, upsamp = test_preamble(est_to_dec, pktdata5)

    est_to_dec2, est_to_int2, pktdata6, re_cfo_2, re_cfo_freq_2, detect = fine_work(pktdata5)
    all_cfo_freq = re_cfo_freq + est_cfo_freq + re_cfo_freq_2
    detect, upsamp = test_preamble(est_to_dec2, pktdata6)
    pktdata7 = pktdata6[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]
    ans2n, ndatas = decode_payload(detect, est_to_dec, pktdata7, pkt_totcnt)

    logger.debug(f'decoded data {len(ans2n)=} {cp_str(ans2n)}')

    payload_data = ndatas[detect + Config.sfdpos + 4:]
    angles = calc_angles(payload_data)
    if opts.debug:
        # noinspection PyArgumentList
        fig1 = px.scatter(y=tocpu(angles), mode="markers")
        fig1.show()
        fig1.write_html(os.path.join(Config.figpath, f'temp_sf7_{pkt_totcnt}.html'))
        for i in range(min(len(ans2n), len(angles))):
            if abs(angles[i]) > 0.5:
                logger.debug(f'abs(angles[i]) > 0.5: {i=} {cp_str(ans2n[i])} {cp_str(angles[i])}')

    return angles


def calc_angles(payload_data):
    angles = cp.zeros(len(payload_data), dtype=float)
    for idx, dataY in enumerate(payload_data):
        dataX = cp.array(dataY)
        data1 = cp.matmul(Config.dataE1, dataX)
        data2 = cp.matmul(Config.dataE2, dataX)
        vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
        est = cp.argmax(vals)
        if est > 0:
            diff_avg0 = cmath.phase(data2[est] / data1[est])
            angles[idx] = diff_avg0
        else:
            angles[idx] = 0
    return angles


def decode_payload(detect, est_to_dec, pktdata4, pkt_totcnt):
    symb_cnt = len(pktdata4) // Config.nsamp
    ndatas = pktdata4[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)
    ans1, power1 = dechirp(ndatas[detect + Config.preamble_len: detect + Config.preamble_len + 2], Config.downchirp)
    ans2, power2 = dechirp(ndatas[detect + Config.sfdpos + 2:], Config.downchirp)
    ans1 = cp.concatenate((ans1, ans2), axis=0)
    power1 = cp.concatenate((power1, power2), axis=0)
    ans1n = ans1
    ans1n += est_to_dec / 8
    if not min(power1) > cp.mean(power1) / 2:
        drop_idx = next((idx for idx, num in enumerate(power1) if num < cp.mean(power1) / 2), -1)
        ans1n = ans1n[:drop_idx]
        logger.info(f'decode: {pkt_totcnt=} power1 drops: {drop_idx=} {len(ans1n)=} {cp_str(power1)}')
    logger.info(f'decode: {len(ans1n)=} {cp_str(ans1n)=} {cp_str(power1)=}')
    return ans1n, ndatas


def test_preamble(est_to_dec, pktdata3):
    ndatas2a = pktdata3[: Config.sfdend * Config.nsamp].reshape(Config.sfdend, Config.nsamp)

    upsamp = Config.fft_upsamp
    detect = 0
    ans1, power1 = dechirp(ndatas2a[detect: detect + Config.preamble_len], Config.downchirp, upsamp)
    ans1 += est_to_dec / 8
    ans2, power2 = dechirp(ndatas2a[detect + Config.sfdpos: detect + Config.sfdpos + 2], Config.upchirp, upsamp)
    ans2 += est_to_dec / 8
    logger.info(f'preamble: {cp_str(ans1)} sfd {cp_str(ans2)} {est_to_dec / 8}')
    logger.info('power' + cp_str(power1, precision=2))

    ans3, phase = dechirp_phase(ndatas2a[detect: detect + Config.preamble_len], Config.downchirp, upsamp)
    logger.info(f'preamble: {cp_str(ans3)} phase: {cp_str(cp.angle(phase))}')
    return detect, upsamp


def fine_work(pktdata2a):
    fft_n = Config.nsamp * Config.fft_upsamp
    symb_cnt = Config.sfdpos + 5  # len(pktdata)//Config.nsamp
    ndatas = pktdata2a[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)
    upsamp = Config.fft_upsamp
    ans1, power1 = dechirp(ndatas, Config.downchirp, upsamp)
    ans2, power2 = dechirp(ndatas, Config.upchirp, upsamp)
    vals = cp.zeros((symb_cnt,), dtype=float)
    for i in range(symb_cnt - (Config.sfdpos + 2)):
        power = cp.sum(power1[i: i + Config.preamble_len]) + cp.sum(power2[i + Config.sfdpos: i + Config.sfdpos + 2])
        ans = cp.abs(cp.sum(cp.exp(1j * 2 * cp.pi / Config.n_classes * ans1[i: i + Config.preamble_len])))
        vals[i] = power * ans
    detect = cp.argmax(vals)
    ansval = average_modulus(ans1[detect: detect + Config.preamble_len], Config.n_classes)
    sfd_upcode = ansval
    ansval2 = average_modulus(ans2[detect + Config.sfdpos: detect + Config.sfdpos + 2], Config.n_classes)
    sfd_downcode = ansval2
    re_cfo_0 = average_modulus((sfd_upcode, sfd_downcode), Config.n_classes)
    est_to_0 = average_modulus((sfd_upcode, - sfd_downcode), Config.n_classes)
    # if opts.debug:
    logger.debug(f'fine work {cp_str(ans1[: Config.preamble_len])} sfd {cp_str(ans2[Config.sfdpos: Config.sfdpos + 2])}'
                 f' {sfd_upcode=}, {sfd_downcode=}, {re_cfo_0=}, {est_to_0=}, {detect=}')
    logger.debug('fine work angles: preamble')
    for sig in ndatas[detect: detect + Config.preamble_len]:
        chirp_data = sig * Config.downchirp
        upsamp = Config.fft_upsamp
        fft_raw = myfft(chirp_data, n=Config.nsamp * upsamp, plan=Config.plans[upsamp])
        target_nfft = Config.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans = round(tocpu(cp.argmax(dat)) / upsamp)

        logger.debug(f'{cmath.phase(cut1[ans])=}, {cmath.phase(cut2[ans])=}')

    re_cfo_freq = re_cfo_0 * (Config.fs / fft_n)
    est_to_int = cp.around(est_to_0)
    est_to_dec = est_to_0 - est_to_int
    pktdata3 = add_freq(pktdata2a, - re_cfo_freq)
    pktdata3 = cp.roll(pktdata3, - tocpu(est_to_int))
    return est_to_dec, est_to_int, pktdata3, re_cfo_0, re_cfo_freq, detect


def gen_upchirp(t0, td, f0, beta):
    # start from ceil(t0in), end
    # logger.debug(f"D {t0=} {td=} {f0=} {beta=}")
    t = (cp.arange(math.ceil(t0), math.ceil(t0 + td), dtype=float) - t0)
    # logger.debug(f"D {t[0]=} {t[-1]=}")
    phase = 2 * cp.pi * (f0 * t + 0.5 * beta * t * t) / Config.fs
    sig = cp.exp(1j * phase)
    return sig


def gen_upchirp_de(t0, td, f0, beta):
    # start from ceil(t0in), end
    # logger.debug(f"D {t0=} {td=} {f0=} {beta=}")
    t = (cp.arange(math.ceil(t0), math.ceil(t0 + td), dtype=float) - t0)
    # logger.debug(f"D {t[0]=} {t[-1]=}")
    sig = cp.exp(2j * cp.pi * (f0 * t + 0.5 * beta * t * t) / Config.fs) * 1j * cp.pi * (f0 + beta * t ) / Config.fs
    return sig


def fine_work_new(pktidx, pktdata2a):
    pktdata2a = togpu(pktdata2a)

    if Config.s1:
        phase = cp.angle(pktdata2a)
        unwrapped_phase = cp.unwrap(phase)
        fig = px.line(y=tocpu(unwrapped_phase[:15 * Config.nsamp]), title="input data 15 symbol")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} input_data.html"))

    # Perform optimization
    def objective(params):
        cfofreq, time_error = params
        pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
        detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error), deadzone=20)
        # tid_times = gen_refchirp_time(cfofreq, time_error - math.ceil(time_error))
        # tid_times_ceil = cp.ceil(tid_times).astype(int)
        res = cp.zeros(len(detect_symb), dtype=cp.complex64)
        ddx = 0 # TODO
        for sidx, ssymb in enumerate(detect_symb):
            ress = cp.conj(ssymb).dot(pktdata2a_roll[ddx : ddx + len(ssymb)])
            ddx += len(ssymb)
            res[sidx] = ress / len(ssymb)
        return - tocpu(cp.sum(cp.abs(res) ** 2))  # Negative because we use a minimizer

    if parse_opts.searchphase:
        if parse_opts.searchfromzero:
            t_lower, t_upper = 0, Config.nsamp
            f_lower, f_upper = -60000, 60000
            bestx = None
            bestobj = cp.inf
        else:
            f_guess = -39500.110 # best for 50/sf7/20
            t_guess = 225.221
            # t_lower, t_upper = t_guess - 50, t_guess + 50
            t_lower, t_upper = 0, Config.nsamp
            f_lower, f_upper = f_guess - 500, f_guess + 500
            bestx = [f_guess, t_guess]
            bestobj = objective(bestx)
        for tryidx in tqdm(range(parse_opts.searchphase_step)):
            start_t = random.uniform(t_lower, t_upper)
            start_f = random.uniform(f_lower, f_upper)
            # noinspection PyTypeChecker
            result = opt.minimize(objective, [start_f, start_t], bounds=[(f_lower, f_upper), (t_lower, t_upper)], method='L-BFGS-B',
                                  options={'gtol': 1e-12, 'disp': False}
                                  )

            if result.fun < bestobj:
                logger.debug(f"{tryidx=: 6d} cfo_freq_est = {result.x[0]:.3f}, time_error = {result.x[1]:.3f} {result.fun=:.3f}")
                bestx = result.x
                # f_guess, t_guess = result.x
                bestobj = result.fun
        cfo_freq_est, time_error = bestx
        logger.info(f"Optimized parameters:\n{cfo_freq_est=}\n{time_error=}")
    else:
        cfo_freq_est = -39500
        time_error = 500+5402 # best for 50/sf7/20
        # cfo_freq_est = -26789.411976307307
        # time_error = 224.64248426804352
        # cfo_freq_est = -25364.299
        # time_error = 922.660
        cfo_freq_est_delta = 0
        time_error_delta = 0
        cfo_freq_est += cfo_freq_est_delta
        time_error += time_error_delta
    if parse_opts.plotmap:
        start_t = np.linspace(0, Config.nsamp, Config.nsamp * 5)
        start_f = np.linspace(-24000, -29000, 1000)
        Z = np.zeros((len(start_f), len(start_t)))
        for i, f in tqdm(enumerate(start_f), total=len(start_f)):
            for j, t in enumerate(start_t):
                Z[i, j] = objective((f, t))

        fig = go.Figure(data=go.Heatmap( z=Z, x=start_t, y=start_f, colorscale='Viridis' ))
        fig.update_layout( title='Heatmap of objective(start_t, start_f)', xaxis_title='t', yaxis_title='f')
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} Plotmap.html"))
        maxidx = np.unravel_index(np.argmin(Z, axis=None), Z.shape, order='C')
        best_f = start_t[maxidx[0]]
        best_t = start_t[maxidx[1]]
        logger.info(f'PlotMap {objective((best_f, best_t))=} {np.min(Z)=} {best_f=} {best_t=}')
        sys.exit(0)
        cfo_freq_est_delta = 0  # 100
        time_error_delta = 0


    if parse_opts.plotline:
        xt_data = np.linspace(time_error - 200, time_error + 200, 1000)
        yval = np.zeros(len(xt_data))
        yval2 = np.zeros(len(xt_data))
        for idx, time_error_2 in enumerate(xt_data):
            detect_symb_p = gen_refchirp(cfo_freq_est, time_error_2 - math.ceil(time_error_2))
            detect_symb_p2 = gen_refchirp_de(cfo_freq_est, time_error_2 - math.ceil(time_error_2))
            didx = math.ceil(time_error_2)
            ssum = 0
            ssum2 = 0
            for sidx, ssymb in enumerate(detect_symb_p[:Config.preamble_len]):
                ress = cp.conj(togpu(ssymb)).dot(pktdata2a[didx: didx + len(ssymb)])
                ssum += cp.abs(ress) ** 2
                ress2 = cp.conj(togpu(detect_symb_p2[sidx])).dot(pktdata2a[didx: didx + len(ssymb)])
                ssum2 += cp.abs(ress2) ** 2
                didx += len(ssymb)
            yval[idx] = ssum
            yval2[idx] = ssum2

        # def quadratic(x, a, b, c):
        #     return a * x ** 2 + b * x + c
        # params, covariance = curve_fit(quadratic, xt_data, yval)
        # logger.info(f"FT fit time line curve {params=}")
        fig = px.line(x = xt_data, y=yval / np.max(yval), title="power with time err")
        # fig.add_trace(go.Scatter(x=xt_data, y=quadratic(xt_data, *params), mode='lines', line=dict(color='red', dash='dash'), name='Fitted Curve'))
        fig.add_trace(go.Scatter(x=xt_data, y=yval2 / np.max(yval2), mode='lines', line=dict(color='red', dash='dash'), name='Derivative'))
        fig.add_vline(x=time_error, line=dict(color='black', width=2, dash='dash'), annotation_text='est_time',
                              annotation_position='top')
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} power with time err.html"))

    pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
    tstart = time_error - math.ceil(time_error) + (Config.sfdend - 0.75) * Config.tsig * (1 - cfo_freq_est / Config.sig_freq)
    if Config.s2:
        detect_symb_plt = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
        detect_symb_plt = cp.concatenate(detect_symb_plt)
        # detect_symb_plt *= (pktdata2a_roll[0] / cp.abs(pktdata2a_roll[0]))
        phase1 = cp.angle(pktdata2a_roll)
        xval = cp.arange(len(detect_symb_plt))
        # xval = cp.arange(len(detect_symb_plt))
        yval1 = cp.unwrap(phase1)
        yval2 = cp.unwrap(cp.angle(detect_symb_plt))
        tsfd = time_error - math.ceil(time_error) + Config.sfdpos * Config.tsig * (1 - cfo_freq_est / Config.sig_freq)
        yval2[math.ceil(tsfd):] += (yval1[math.ceil(tsfd)] - yval2[math.ceil(tsfd)])

        fig = go.Figure()
        # view_len = 60
        fig.add_trace(go.Scatter(x=tocpu(xval), y=tocpu(yval1[xval]), mode='lines', name='input', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=tocpu(xval), y=tocpu(yval2[xval]), mode='lines', name='fit', line=dict(dash='dash', color='red')))
        # view_len = 60
        # fig.add_trace(go.Scatter(x=tocpu(xval)[:Config.nsamp * view_len], y=tocpu(yval1[xval])[:Config.nsamp * view_len], mode='lines', name='input', line=dict(color='blue')))
        # fig.add_trace(go.Scatter(x=tocpu(xval)[:Config.nsamp * view_len], y=tocpu(yval2[xval])[:Config.nsamp * view_len], mode='lines', name='fit', line=dict(dash='dash', color='red')))
        # fig.add_trace(go.Scatter(x=tocpu(xval)[:Config.nsamp * view_len], y=tocpu(yval2[xval])[:Config.nsamp * view_len], mode='lines', name='fit', line=dict(color='red')))
        fig.update_layout(title='aligned pkt', legend=dict(x=0.1, y=1.1))
        if not parse_opts.noplot: fig.show()

    if parse_opts.end1: sys.exit(0)

    if Config.s3:
        detect_symb_p = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
        didx = 0
        res_angle = cp.zeros((Config.preamble_len,), dtype=float)
        for sidx, ssymb in enumerate(detect_symb_p[:Config.preamble_len]):
            ress = cp.conj(togpu(ssymb)).dot(pktdata2a_roll[didx: didx + len(ssymb)])
            rangle = cp.angle(ress)
            res_angle[sidx] = rangle
            # logger.info(f'{rangle=}')
            didx += len(ssymb)

        def quadratic(x, a, b, c):
            return a * x ** 2 + b * x + c

        x_data = np.arange(len(res_angle), dtype=float)
        res_angle = cp.unwrap(res_angle)
        # noinspection PyTupleAssignmentBalance
        params, covariance = curve_fit(quadratic, x_data, tocpu(res_angle))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=tocpu(res_angle), mode='markers', name='Input Data'))
        fig.add_trace(go.Scatter(x=x_data, y=quadratic(x_data, *params), mode="lines", name='Fitted Curve'))
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} res_angle.html"))

    if parse_opts.savefile:
        pktdatas = pktdata2a_roll[math.ceil(tstart):]
        return pktdatas, tstart - math.ceil(tstart), cfo_freq_est

    code_cnt = math.floor(len(pktdata2a_roll) / Config.nsamp - Config.sfdend - 0.5)
    logger.info(f"{code_cnt=}")

    tstart_p = tstart - math.ceil(tstart)
    pktdatas = pktdata2a_roll[math.ceil(tstart):]
    decode_new(pktidx, pktdatas, tstart_p, cfo_freq_est)
    return pktdatas, tstart_p, cfo_freq_est

def decode_new(pktidx, pktdatas, tstart_p, cfo_freq_est):
    code_cnt = math.floor(len(pktdatas) / Config.nsamp)
    sigt = Config.tsig * (1 - cfo_freq_est / Config.sig_freq)
    tid_times = (cp.arange(code_cnt + 1, dtype=float)) * sigt + tstart_p

    code_ests = cp.zeros((code_cnt,), dtype=int)
    code_ests_LT = cp.zeros((code_cnt,), dtype=int)
    code_powers = cp.zeros((code_cnt,), dtype=int)
    angle1 = cp.zeros((code_cnt,), dtype=float)
    angle2 = cp.zeros((code_cnt,), dtype=float)
    angle3 = cp.zeros((code_cnt,), dtype=float)
    angle4 = cp.zeros((code_cnt,), dtype=float)
    beta = Config.bw / sigt
    tsig_arr = sigt * (1 - cp.arange(Config.n_classes, dtype=float) / Config.n_classes)

    for codeid in tqdm(range(code_cnt)):
        upchirp1_arr = [gen_upchirp(tstart_p + sigt * codeid, sigt * (1 - code / Config.n_classes),
                                    ((code / Config.n_classes - 0.5) * Config.bw) + cfo_freq_est, beta)
                        for code in range(Config.n_classes)]
        upchirp2_arr = [gen_upchirp(tstart_p + sigt * (codeid + 1 - code / Config.n_classes), sigt * code / Config.n_classes,
                                    -Config.bw / 2 + cfo_freq_est, beta) if code != 0 else None
                        for code in range(Config.n_classes)]
        res1_arr = cp.zeros(Config.n_classes, dtype=cp.complex64)
        res2_arr = cp.zeros(Config.n_classes, dtype=cp.complex64)
        for code in range(Config.n_classes):
            res1_arr[code] = cp.conj(upchirp1_arr[code]).dot(pktdatas[math.ceil(tid_times[codeid]): math.ceil(tid_times[codeid] + tsig_arr[code])]) #/ tsig_arr[code]
        for code in range(1, Config.n_classes):
            res2_arr[code] = cp.conj(upchirp2_arr[code]).dot(pktdatas[math.ceil(tid_times[codeid] + tsig_arr[code]): math.ceil(tid_times[codeid + 1])]) # / (tid_times[tid + 1] - tid_times[tid] - tsig_arr[code])
        res_array = cp.abs(res1_arr) ** 2 + cp.abs(res2_arr) ** 2
        est_code = tocpu(cp.argmax(res_array))
        code_powers[codeid] = cp.max(res_array)
        logger.debug(f"L curvefit {est_code=} maxval={tocpu(cp.max(res_array))}")
        code_ests[codeid] = est_code
        if Config.s4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=tocpu(res_array), mode='lines+markers'))
            fig.add_vline(x=est_code, line=dict(color='black', width=2, dash='dash'), annotation_text='est_code',
                          annotation_position='top')
            fig.update_layout(title=f"resarray {codeid=} {est_code=}")
            if not parse_opts.noplot: fig.show()
            fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} resarray {codeid=} {est_code=}.html"))

            # phase = cp.angle(pktdata2a_roll[math.ceil(tid_times[tid]): math.ceil(tid_times[tid + 1])])
            # unwrapped_phase = cp.unwrap(phase)
            # fig = px.line(y=tocpu(unwrapped_phase), title=f"phase {tid=} {est_code=}")
            # if not parse_opts.noplot: fig.show()
            # fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} phase {tid=} {est_code=}.html"))

        if Config.s7 or parse_opts.compLTSNR:
            dataX = pktdatas[math.ceil(tid_times[codeid]): math.ceil(tid_times[codeid]) + Config.nsamp]
            dataX = add_freq(dataX, - cfo_freq_est)
            dataX = dataX.T
            data1 = cp.matmul(Config.dataE1, dataX)
            data2 = cp.matmul(Config.dataE2, dataX)
            vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
            est = tocpu(cp.argmax(vals))
            code_ests_LT[codeid] = est
            if Config.s7:
                angle1[codeid] = cp.angle(res1_arr[est_code])
                angle2[codeid] = cp.angle(res2_arr[est_code])
                # logger.info(f"abs1={cp.abs(res1_arr[est_code])} angle1={cp.angle(res1_arr[est_code])} abs2={cp.abs(res2_arr[est_code])} angle2={cp.angle(res2_arr[est_code])} {est_code=} tot_power={cp.abs(res1_arr[est_code])**2+cp.abs(res2_arr[est_code])**2}")

                angle3[codeid] = cp.angle(data1[est])
                angle4[codeid] = cp.angle(data2[est])
                # logger.info(f"LT: {tid=} {est=} {cp.abs(data1[est])=} {cp.abs(data2[est])=} {angle3[tid]=} {angle4[tid]=}")
        if Config.sflag_end is not None and codeid >= code_cnt - Config.sflag_end:
            Config.s5 = 1
            Config.s6 = 1
        else:
            Config.s5 = 0
            Config.s6 = 0
        if Config.s5 or Config.s6:
            upchirp_est = upchirp1_arr[est_code] * res1_arr[est_code] / cp.abs(res1_arr[est_code])
            if est_code != 0:
                upchirp2_est = upchirp2_arr[est_code] * res2_arr[est_code] / cp.abs(res2_arr[est_code])
                upchirp_est = cp.concatenate((upchirp_est, upchirp2_est))
            sigtt = cp.arange(math.ceil(tid_times[codeid]), math.ceil(tid_times[codeid + 1]), dtype=int)
            phase1 = cp.angle(pktdatas[math.ceil(tid_times[codeid]): math.ceil(tid_times[codeid + 1])])
            # logger.info(f"P1 {tid_times[tid]=} {tid=} {sigt=}")
            phase2 = cp.angle(upchirp_est)
        if Config.s5:
            fig = px.line(x=tocpu(sigtt), y=[tocpu(cp.unwrap(phase1)), tocpu(cp.unwrap(phase2))], color_discrete_sequence=['blue', 'red'], title=f"fit code {codeid=} {est_code=}")
            fig.data[0].name = 'Input'
            fig.data[1].name = 'Fitting'
            fig.data[1].line = dict(dash='dash')
            fig.add_vline(x=tid_times[codeid] + tsig_arr[est_code], line=dict(color='black', width=2, dash='dash'), annotation_text='est_code',
                          annotation_position='top')
            if not parse_opts.noplot: fig.show()
            fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} fit code {codeid=} {est_code=}.html"))
        if Config.s6:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=tocpu(cp.diff(cp.unwrap(phase1))), mode='lines', name='Input Data'))
            fig.add_trace(go.Scatter(y=tocpu(cp.diff(cp.unwrap(phase2))), mode="lines", line=dict(dash='dash', color='red'), name='Fitted Curve'))
            # fig.add_vline(x=tid_times[tid] + tsig_arr[est_code], line=dict(color='black', width=2, dash='dash'), annotation_text='est_code',
            #               annotation_position='top')
            if not parse_opts.noplot: fig.show()
            fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} fit code2 {codeid=} {est_code=}.html"))
    if Config.s7:
        fig = px.scatter(y=tocpu(code_ests), title=f"pkt{pktidx} code estimations")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} code estsyy.html"))

        fig = px.scatter(y=tocpu(code_powers), title=f"pkt{pktidx} code powers")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} code powers.html"))

        fig = px.scatter(x=tocpu(code_ests), y=tocpu(code_powers), title=f"pkt{pktidx} code powers to code")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} code powers to code.html"))

        fig = px.line(y=[tocpu(angle1), tocpu(angle2)], color_discrete_sequence=['blue', 'red'],
                      title=f"pkt{pktidx} angles of symbols")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} angles of symbols.html"))

        # noinspection PyArgumentList
        fig = px.scatter(x=tocpu(code_ests), y=[tocpu(angle1), tocpu(angle2)], color_discrete_sequence=['blue', 'red'],
                      title=f"pkt{pktidx} angles of symbols vs code")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} angles of symbols vs code.html"))

        fig = px.line(y=[tocpu(angle3), tocpu(angle4)], color_discrete_sequence=['blue', 'red'],
                      title=f"pkt{pktidx} angles of symbols LT method")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} angles of symbols LT method.html"))

        # noinspection PyArgumentList
        fig = px.scatter(x=tocpu(code_ests), y=[tocpu(angle3), tocpu(angle4)], color_discrete_sequence=['blue', 'red'],
                         title=f"pkt{pktidx} angles of symbols vs code LT method")
        if not parse_opts.noplot: fig.show()
        fig.write_html(os.path.join(Config.figpath, f"pkt{pktidx} angles of symbols vs code LT method.html"))
    return code_ests, code_ests_LT


def gen_refchirp_time(cfofreq, tstart):
    tind_times = cp.arange(Config.sfdend + 1, dtype=float)
    tind_times[-1] -= 0.75
    tid_times = tind_times * Config.tsig * (1 - cfofreq / Config.sig_freq) + tstart
    return tid_times


def gen_refchirp(cfofreq, tstart, deadzone=0):
    detect_symb = []
    # tid_times = gen_refchirp_time(cfofreq, tstart)
    sigt = Config.tsig * (1 + cfofreq / Config.sig_freq)
    beta = Config.bw / Config.tsig
    for tid in range(Config.preamble_len):
        upchirp = gen_upchirp(tstart + sigt * tid, sigt, -Config.bw / 2 + cfofreq, beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    for tid in range(Config.preamble_len, Config.sfdpos):
        detect_symb.append(cp.zeros(math.ceil(tstart + sigt * (tid + 1)) - math.ceil(tstart + sigt * tid), dtype=cp.complex64))
    for tid in range(Config.sfdpos, Config.sfdend):
        upchirp = gen_upchirp(tstart + sigt * tid, sigt if tid != Config.sfdend - 1 else sigt / 4,
                              Config.bw / 2 + cfofreq, - beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    return detect_symb

def gen_refchirp_de(cfofreq, tstart, deadzone=0):
    detect_symb = []
    # tid_times = gen_refchirp_time(cfofreq, tstart)
    sigt = Config.tsig * (1 - cfofreq / Config.sig_freq)
    beta = Config.bw / sigt
    for tid in range(Config.preamble_len):
        upchirp = gen_upchirp_de(tstart + sigt * tid, sigt, -Config.bw / 2 + cfofreq, beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    for tid in range(Config.preamble_len, Config.sfdpos):
        detect_symb.append(cp.zeros(math.ceil(tstart + sigt * (tid + 1)) - math.ceil(tstart + sigt * tid), dtype=cp.complex64))
    for tid in range(Config.sfdpos, Config.sfdend):
        upchirp = gen_upchirp_de(tstart + sigt * tid, sigt if tid != Config.sfdend - 1 else sigt / 4,
                              Config.bw / 2 + cfofreq, - beta)
        # assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        if deadzone > 0:
            upchirp[:deadzone] = cp.zeros(deadzone, dtype=cp.complex64)
            upchirp[-deadzone:] = cp.zeros(deadzone, dtype=cp.complex64)
        detect_symb.append(upchirp)
    return detect_symb


def read_large_file(file_path_in):
    with open(file_path_in, 'rb') as file:
        while True:
            try:
                rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp)
            except EOFError:
                logger.warning("file complete with EOF")
                break
            if len(rawdata) < Config.nsamp:
                logger.warning(f"file complete, {len(rawdata)=}")
                break
            yield rawdata


def read_pkt(file_path_in, threshold, min_length=20):
    current_sequence = []
    for rawdata in read_large_file(file_path_in):
        number = cp.max(cp.abs(rawdata))
        if number > threshold:
            current_sequence.append(rawdata)
        else:
            if len(current_sequence) > min_length:
                yield cp.concatenate(current_sequence)
            current_sequence = []


# read packets from file
if __name__ == "__main__":
    angles = []

    if parse_opts.fromfile:
        with open(f"{parse_opts.fromfile}", "rb") as f:
            pktdata_lst, tstart_lst, cfo_freq_est = pickle.load(f)
            code_ref = cp.array([33,61,61,113,73,25,29,45,51,102,75,86,110,43,91,78,10,109,39,109,76,103,110,10,90,52,40,80,6,12,118,68,104,79,99,59,120,23,84,64,123,11,108,55,20,48,69,102,25,50,29,40,96,71,115,86,4,8,113,63,125,2,1,52,4,8,16,32,100,58,28,62,105,48,32,32,69,122,109,23,97,63,68,71,114,22,81,90,126,6,117,10,43,93,116,31,56,18,29,25,39,76,85,122,62,124,55,77,111,46,95,0,7,116,26,20,57,0,3,1,7,13,25,49,9,17,59,27,28,56,17,64,93,79,18,89,114,29,121,47,86,91,120,32,60,10,45,121,55,108,33,78,3,65,32,33,32,16,3,0])
            if parse_opts.decode_unknown: code_ref = None
            snr_range = np.arange(parse_opts.snrlow, parse_opts.snrhigh, dtype=int)
            snr_list = np.zeros((2, len(snr_range)))
            snr_list_cnt = 0
            logger.info(f"{len(snr_range)=} {snr_list.shape=}")
            for idx, (p, t, c) in enumerate(zip(pktdata_lst, tstart_lst, cfo_freq_est)):
                if parse_opts.compLTSNR:
                    if code_ref is None:
                        code_ref, code_ref_LT = decode_new(idx, p, t, c)
                        logger.info(f"decoderef: ACC:{tocpu(cp.sum(code_ref_LT[:lenc]==code_ref[:lenc])/lenc)}")
                        logger.info(f"ref:{cp_str(code_ref, precision=0)}")
                    else:
                        code_res, code_res_LT = decode_new(idx, p, t, c)
                        lenc = min(len(code_res), len(code_ref))
                        if cp.sum(code_ref[:lenc]==code_res[:lenc]) != lenc:
                            logger.warning(f"skipping: decodecheck ACC:{tocpu(cp.sum(code_ref[:lenc] == code_res[:lenc]) / lenc)}")
                            continue
                        if cp.sum(code_ref[:lenc] == code_res_LT[:lenc]) != lenc:
                            logger.warning(f"skipping: decodecheck ACC:{tocpu(cp.sum(code_ref[:lenc] == code_res_LT[:lenc]) / lenc)}")
                            continue
                    snr_list_cnt += 1
                    for snridx, snr in enumerate(snr_range):
                        amp = math.pow(0.1, snr / 20) * np.mean(np.abs(p))
                        noise = amp / math.sqrt(2) * cp.random.randn(len(p)) + 1j * amp / math.sqrt(2) * cp.random.randn(len(p))
                        pX = p + togpu(noise)  # dataX: data with noise
                        code_res, code_res_LT = decode_new(idx, pX, t, c)
                        lenc = min(len(code_res), len(code_ref))
                        acc = tocpu(cp.sum(code_res[:lenc] == code_ref[:lenc]) / lenc)
                        snr_list[0, snridx] += acc
                        acc_LT = tocpu(cp.sum(code_res_LT[:lenc] == code_ref[:lenc]) / lenc)
                        snr_list[1, snridx] += acc_LT
                        logger.info(f"decode:{idx} len:{len(code_res)}/{len(code_ref)} SNR:{snr} ACC:{acc} ACCLT:{acc_LT}")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=snr_range, y=snr_list[0]/snr_list_cnt, mode='lines', name='Ours'))
                        fig.add_trace(go.Scatter(x=snr_range, y=snr_list[1]/snr_list_cnt, mode='lines', name='LT'))
                        fig.write_html(os.path.join(Config.figpath, f"acc_snr_ours_LT.html"))
                        if not parse_opts.noplot: fig.show()
                    logger.info(f"snrlow:{parse_opts.snrlow} acc:{cp_str(snr_list/snr_list_cnt)}")
                else:
                    code_res, _ = decode_new(idx, p, t, c)
                    if code_ref is None:
                        code_ref = code_res
                        logger.info(f"ref:{cp_str(code_ref, precision=0)}")
                    else:
                        lenc = min(len(code_res), len(code_ref))
                        logger.info(f"decode:{idx} len:{len(code_res)}/{len(code_ref)} ACC:{tocpu(cp.sum(code_res[:lenc]==code_ref[:lenc])/lenc)}")
            if parse_opts.compLTSNR:
                logger.info(f"compLTSNR fin snrlow:{parse_opts.snrlow} acc:{cp_str(snr_list/snr_list_cnt)}")
    else:
        for file_path in Config.file_paths:
            pkt_cnt = 0
            pktdata = []
            fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
            logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')

            power_eval_len = 5000
            nmaxs = cp.zeros(power_eval_len, dtype=float)
            for idx, rawdata in enumerate(read_large_file(file_path)):
                nmaxs[idx] = cp.max(cp.abs(rawdata))
                if idx == power_eval_len - 1: break
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            kmeans.fit(tocpu(nmaxs.reshape(-1, 1)))
            thresh = cp.mean(kmeans.cluster_centers_)
            counts, bins = cp.histogram(nmaxs, bins=100)
            # logger.debug(f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")
            logger.debug(f"cluster: {kmeans.cluster_centers_[0]} {kmeans.cluster_centers_[1]} {thresh=}")
            threshpos = np.searchsorted(tocpu(bins), thresh).item()
            logger.debug(f"lower: {cp_str(counts[:threshpos])}")
            logger.debug(f"higher: {cp_str(counts[threshpos:])}")

            pkt_totcnt = 0
            pktdata_lst = []
            tstart_lst = []
            cfo_freq_est = []
            for pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=20)):
                if pkt_idx == 0: continue
                logger.info(f"Prework {pkt_idx=} {len(pkt_data)=}")
                p, t, c = fine_work_new(pkt_idx, pkt_data / cp.mean(cp.abs(pkt_data)))
                pktdata_lst.append(p)
                tstart_lst.append(t)
                cfo_freq_est.append(c)
                with open(f"dataout{parse_opts.searchphase_step}.pkl","wb") as f:
                    pickle.dump((pktdata_lst, tstart_lst, cfo_freq_est),f)
