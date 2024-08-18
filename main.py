import argparse
import logging
import os
import random
import sys
import time

import cmath
import math
# import matplotlib.pyplot as plt, mpld3
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
# Enable fallback mode
# from cupyx.fallback_mode import numpy as np
import numpy as np
import scipy.optimize as opt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from tqdm import tqdm
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', default=False, help='Use cpu instead of gpu (numpy instead of cupy)')
parser.add_argument('--searchphase', action='store_true', default=False)
parser.add_argument('--plotmap', action='store_true', default=False)
parser.add_argument('--end1', action='store_true', default=False)
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
logger2 = logging.getLogger('debug_logger')

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

    sf = 7
    bw = 125e3
    fs = 1e6
    sig_freq = 470e6
    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    # base_dir = '/data/djl/datasets/Dataset_50Nodes'
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)
    file_paths = ['/data/djl/datasets/Dataset_50Nodes/sf7-470-new-70.bin']
    # file_paths = ['/data/djl/datasets/sf7-470-pre-2.bin']
    # file_paths = ['/data/djl/datasets/sf7-470-pre-2.bin']
    # for file_name in os.listdir(base_dir):
    #     if file_name.startswith('sf7') and file_name.endswith('.bin'):
    #         file_paths.append(os.path.join(base_dir, file_name))

    nsamp = round(n_classes * fs / bw)
    nfreq = 1024 + 1
    time_upsamp = 32

    preamble_len = 8  # TODO
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
    pktdata_in /= cp.mean(cp.abs(pktdata_in))
    time_error, cfo_freq_est = fine_work_new(pktdata_in)

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
    logger2.debug(f"{t0=} {td=} {f0=} {beta=}")
    assert -1 <= t0 <= 0
    t = (cp.arange(math.floor(t0 + td), dtype=float) - t0) / Config.fs
    logger2.debug(f"{t[0]=} {t[-1]=}")
    phase = 2 * cp.pi * (f0 * t + 0.5 * beta * t * t)
    sig = cp.exp(1j * phase)
    return sig


def fine_work_new(pktdata2a):  # TODO working
    pktdata2a = togpu(pktdata2a)

    phase = cp.angle(pktdata2a)
    unwrapped_phase = cp.unwrap(phase)
    fig = px.line(y=tocpu(unwrapped_phase[:15 * 1024]), title="input data 15 symbol")
    fig.show()
    fig.write_html(os.path.join(Config.figpath, f"input_data.html"))

    # Perform optimization
    if parse_opts.searchphase:
        def objective(params):
            cfofreq, time_error = params
            pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
            detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error), deadzone=20)
            tid_times = gen_refchirp_time(cfofreq, time_error - math.ceil(time_error))
            tid_times_ceil = cp.ceil(tid_times).astype(int)
            res = cp.zeros(len(detect_symb), dtype=cp.complex64)
            for sidx, ssymb in enumerate(detect_symb):
                ress = cp.conj(ssymb).dot(pktdata2a_roll[tid_times_ceil[sidx]: tid_times_ceil[sidx + 1]])
                res[sidx] = ress / (tid_times_ceil[sidx + 1] - tid_times_ceil[sidx])
            return - tocpu(cp.sum(cp.abs(res) ** 2))  # Negative because we use a minimizer

        # t_lower, t_upper = 0, Config.nsamp # TODO
        # f_lower, f_upper = -30000, -23000
        # bestx = None
        # bestobj = cp.inf
        f_guess = -25364.299
        t_guess = 922.660
        for tryidx in range(10000):
            t_lower, t_upper = t_guess - 50, t_guess + 50
            f_lower, f_upper = f_guess - 200, f_guess + 200
            bestx = [f_guess, t_guess]
            bestobj = objective(bestx)
            start_t = random.uniform(t_lower, t_upper)
            start_f = random.uniform(f_lower, f_upper)
            # noinspection PyTypeChecker
            result = opt.minimize(objective, [start_f, start_t], bounds=[(f_lower, f_upper), (t_lower, t_upper)], method='L-BFGS-B',
                                  options={'gtol': 1e-8, 'disp': False}
                                  )

            if result.fun < bestobj:
                logger.debug(f"{tryidx=: 6d} cfo_freq_est = {result.x[0]:.3f}, time_error = {result.x[1]:.3f} {result.fun=:.3f}")
                bestx = result.x
                f_guess, t_guess = result.x
                bestobj = result.fun
        cfo_freq_est, time_error = bestx
        logger.info(f"Optimized parameters: {cfo_freq_est=} {time_error=}")

        if parse_opts.plotmap:
            start_t = np.linspace(0, Config.nsamp, Config.nsamp * 5)
            start_f = np.linspace(-24000, -29000, 100)
            Z = np.zeros((len(start_f), len(start_t)))
            for i, f in tqdm(enumerate(start_f), total=len(start_f)):
                for j, t in enumerate(start_t):
                    Z[i, j] = objective((f, t))

            fig = go.Figure(data=go.Heatmap( z=Z, x=start_t, y=start_f, colorscale='Viridis' ))
            fig.update_layout( title='Heatmap of objective(start_t, start_f)', xaxis_title='t', yaxis_title='f')
            fig.show()
            fig.write_html(os.path.join(Config.figpath, f"Plotmap.html"))
            maxidx = np.unravel_index(np.argmin(Z, axis=None), Z.shape, order='C')
            best_f = start_t[maxidx[0]]
            best_t = start_t[maxidx[1]]
            logger.info(f'PlotMap {objective((best_f, best_t))=} {np.min(Z)=} {best_f=} {best_t=}')
            sys.exit(0)
        cfo_freq_est_delta = 0  # 100
        time_error_delta = 0
    else:
        # cfo_freq_est = -26685.110 # best for 50/sf7/20
        # time_error = 225.221
        cfo_freq_est = -25364.299
        time_error = 922.660
        cfo_freq_est_delta = 0  # TODO
        time_error_delta = 0
        cfo_freq_est += cfo_freq_est_delta
        time_error += time_error_delta

    pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
    phase1 = cp.angle(pktdata2a_roll)
    detect_symb_plt = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
    detect_symb_plt = cp.concatenate(detect_symb_plt)
    detect_symb_plt *= (pktdata2a_roll[0] / cp.abs(pktdata2a_roll[0]))
    tstart = time_error - math.ceil(time_error) + (Config.sfdend - 0.75) * Config.tsig * (1 - cfo_freq_est / Config.sig_freq)
    xval = cp.arange(1024*60, 1024*65)
    # xval = cp.arange(len(detect_symb_plt))
    yval1 = cp.unwrap(phase1)
    yval2 = cp.unwrap(cp.angle(detect_symb_plt))
    tsfd = time_error - math.ceil(time_error) + Config.sfdpos * Config.tsig * (1 - cfo_freq_est / Config.sig_freq)
    yval2[math.ceil(tsfd):] += (yval1[math.ceil(tsfd)] - yval2[math.ceil(tsfd)])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tocpu(xval), y=tocpu(yval1[xval]), mode='lines', name='input', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=tocpu(xval), y=tocpu(yval2[xval]), mode='lines', name='fit', line=dict(dash='dash', color='red')))
    fig.update_layout(title='aligned pkt', legend=dict(x=0.1, y=1.1))
    fig.show()

    if parse_opts.end1: sys.exit(0)

    detect_symb_rangle = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
    didx = 0
    res_angle = cp.zeros((Config.preamble_len,), dtype=float)
    for sidx, ssymb in enumerate(detect_symb_rangle[:Config.preamble_len]):
        ress = cp.conj(togpu(ssymb)).dot(pktdata2a_roll[didx: didx + len(ssymb)])
        rangle = cp.angle(ress)
        res_angle[sidx] = rangle
        # logger.info(f'{rangle=}')
        didx += len(ssymb)

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    # res_angle = res_angle[1:-1]  # TODO
    x_data = np.arange(len(res_angle), dtype=float)
    # add_data = cp.array([2,2,1,1,0,0,0,0])
    # res_angle += add_data * cp.pi * 2
    res_angle = cp.unwrap(res_angle)
    # res_angle[-1] += np.pi * 2 # TODO
    # noinspection PyTupleAssignmentBalance
    params, covariance = curve_fit(quadratic, x_data, tocpu(res_angle))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=tocpu(res_angle), mode='markers', name='Input Data'))
    fig.add_trace(go.Scatter(x=x_data, y=quadratic(x_data, *params), mode="lines", name='Fitted Curve'))
    fig.show()
    fig.write_html(os.path.join(Config.figpath, f"res_angle.html"))

    code_cnt = math.floor(len(pktdata2a_roll) / Config.nsamp - Config.sfdend - 0.5)
    code_ests = cp.zeros((code_cnt,), dtype=int)
    angle1 = cp.zeros((code_cnt,), dtype=float)
    angle2 = cp.zeros((code_cnt,), dtype=float)
    angle3 = cp.zeros((code_cnt,), dtype=float)
    angle4 = cp.zeros((code_cnt,), dtype=float)

    sigt = Config.tsig * (1 - cfo_freq_est / Config.sig_freq)
    tstart_p = (Config.sfdpos + 2.25) * sigt + tstart
    beta = Config.bw / sigt
    tid_times = (cp.arange(code_cnt + 1, dtype=float) + Config.sfdpos + 2.25) * sigt + tstart
    tsig_arr = sigt * (1 - cp.arange(Config.n_classes, dtype=float) / Config.n_classes)
    for tid in range(code_cnt):
        upchirp1_arr = [gen_upchirp(tstart_p + sigt * tid, sigt * (1 - code / Config.n_classes),
                                    ((code / Config.n_classes - 0.5) * Config.bw) + cfo_freq_est, beta)
                        for code in range(Config.n_classes)]
        upchirp2_arr = [gen_upchirp(tstart_p + sigt * (tid + 1 - code / Config.n_classes), sigt * code / Config.n_classes,
                                    -Config.bw / 2 + cfo_freq_est, beta) if code != 0 else None
                        for code in range(Config.n_classes)]
        res1_arr = cp.zeros(Config.n_classes, dtype=cp.complex64)
        res2_arr = cp.zeros(Config.n_classes, dtype=cp.complex64)
        for code in range(Config.n_classes):
            res1_arr[code] = cp.conj(upchirp1_arr[code]).dot(pktdata2a_roll[math.ceil(tid_times[tid]): math.ceil(tid_times[tid] + tsig_arr[code])])
        for code in range(1, Config.n_classes):
            res2_arr[code] = cp.conj(upchirp2_arr[code]).dot(pktdata2a_roll[math.ceil(tid_times[tid] + tsig_arr[code]): math.ceil(tid_times[tid + 1])])
        res_array = cp.abs(res1_arr) ** 2 + cp.abs(res2_arr) ** 2
        est_code = tocpu(cp.argmax(res_array))
        logger.info(f"Log curvefit {est_code=} maxval={tocpu(cp.max(res_array))}")
        code_ests[tid] = est_code

        fig = px.line(y=tocpu(res_array), title=f"resarray {tid=} {est_code=}")
        fig.add_vline(x=est_code, line=dict(color='black', width=2, dash='dash'), annotation_text='est_code',
                      annotation_position='top')
        fig.write_html(os.path.join(Config.figpath, f"resarray {tid=} {est_code=}.html"))

        phase = cp.angle(pktdata2a_roll[math.ceil(tid_times[tid]): math.ceil(tid_times[tid + 1])])
        unwrapped_phase = cp.unwrap(phase)
        fig = px.line(y=tocpu(unwrapped_phase), title=f"phase {tid=} {est_code=}")
        fig.show()
        fig.write_html(os.path.join(Config.figpath, f"phase {tid=} {est_code=}.html"))

        angle1[tid] = cp.angle(res1_arr[est_code])
        angle2[tid] = cp.angle(res2_arr[est_code])

        # logger.info(f"{cp.abs(res1)=} {cp.angle(res1)=} {code=} {cp.abs(res1)**2+cp.abs(res2)**2=}")

        dataX = pktdata2a_roll[math.ceil(tid_times[tid]): math.ceil(tid_times[tid]) + Config.nsamp]
        dataX = dataX.T
        data1 = cp.matmul(Config.dataE1, dataX)
        data2 = cp.matmul(Config.dataE2, dataX)
        vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
        est = tocpu(cp.argmax(vals))
        angle3[tid] = cp.angle(data1[est])
        angle4[tid] = cp.angle(data2[est])
        logger.info(f"LT: {tid=} {est=} {cp.abs(data1[est])=} {cp.abs(data2[est])=} {angle3[tid]=} {angle4[tid]=}")

        upchirp_est = upchirp1_arr[est_code] * res1_arr[est_code] / cp.abs(res1_arr[est_code])
        if est_code != 0:
            upchirp2_est = upchirp2_arr[est_code] * res2_arr[est_code] / cp.abs(res2_arr[est_code])
            upchirp_est = cp.concatenate((upchirp_est, upchirp2_est))
        sigtt = cp.arange(math.ceil(tid_times[tid]), math.ceil(tid_times[tid + 1]), dtype=int)
        phase1 = cp.angle(pktdata2a_roll[sigtt])
        phase2 = cp.angle(upchirp_est)
        fig = px.line(x=tocpu(sigtt), y=[tocpu(cp.unwrap(phase1)), tocpu(cp.unwrap(phase2))], color_discrete_sequence=['blue', 'red'], title=f"fit code {tid=} {est_code=}")
        fig.data[0].name = 'Input'
        fig.data[1].name = 'Fitting'
        fig.data[1].line = dict(dash='dash')
        fig.add_vline(x=tid_times[tid] + tsig_arr[est_code], line=dict(color='black', width=2, dash='dash'), annotation_text='est_code',
                      annotation_position='top')
        fig.show()
        fig.write_html(os.path.join(Config.figpath, f"fit code {tid=} {est_code=}.html"))
        sys.exit(0)
    fig = px.line(y=[angle1, angle2], color_discrete_sequence=['blue', 'red'],
                  title=f"angles of symbols")
    fig.write_html(os.path.join(Config.figpath, f"angles of symbols.html"))
    fig = px.scatter(x=code_ests, y=[angle1, angle2], color_discrete_sequence=['blue', 'red'], mode="markers",
                  title=f"angles of symbols vs code")
    fig.write_html(os.path.join(Config.figpath, f"angles of symbols vs code.html"))
    fig = px.line(y=[angle3, angle4], color_discrete_sequence=['blue', 'red'],
                  title=f"angles of symbols LT method")
    fig.write_html(os.path.join(Config.figpath, f"angles of symbols LT method.html"))
    fig = px.scatter(x=code_ests, y=[angle3, angle4], color_discrete_sequence=['blue', 'red'], mode="markers",
                     title=f"angles of symbols vs code LT method")
    fig.write_html(os.path.join(Config.figpath, f"angles of symbols vs code LT method.html"))

    return time_error, cfo_freq_est


def gen_refchirp_time(cfofreq, tstart):
    tind_times = cp.arange(Config.sfdend + 1, dtype=float)
    tind_times[-1] -= 0.75
    tid_times = tind_times * Config.tsig * (1 - cfofreq / Config.sig_freq) + tstart
    return tid_times


def gen_refchirp(cfofreq, tstart, deadzone=0):
    detect_symb = []
    # tid_times = gen_refchirp_time(cfofreq, tstart)
    sigt = Config.tsig * (1 - cfofreq / Config.sig_freq)
    beta = Config.bw / sigt
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
        upchirp = gen_upchirp(tstart + sigt * tid, sigt, Config.bw / 2 + cfofreq, - beta)
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
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=20)):
            if pkt_idx < 20: continue
            logger.info(f"Prework {pkt_idx=} {len(pkt_data)=}")
            work(pkt_idx, pkt_data)
