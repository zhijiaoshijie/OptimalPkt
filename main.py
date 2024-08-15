import logging
import os
import pickle
import sys
import time

import cmath
import math
import matplotlib.pyplot as plt
# Enable fallback mode
# from cupyx.fallback_mode import numpy as np
import numpy as np
import scipy.optimize as opt
from scipy.signal import chirp
from sklearn.cluster import KMeans
from tqdm import tqdm

use_gpu = False
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


def mychirp(t, f0, f1, t1, method, phi):
    return togpu(chirp(tocpu(t), f0, f1, t1, method, phi))


def cp_str(x, precision=2, suppress_small=False):
    return np.array2string(tocpu(x), precision=precision, formatter={'float_kind': lambda x: f"{x:.2f}"}, floatmode='fixed', suppress_small=suppress_small)


def myscatter(x, y, **kwargs):
    plt.scatter(tocpu(x), tocpu(y), **kwargs)


def myplot(*args, **kwargs):
    if len(args) == 1:
        plt.plot(tocpu(args[0]), **kwargs)
    elif len(args) == 2:
        plt.plot(tocpu(args[0]), tocpu(args[1]), **kwargs)
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
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

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
    base_dir = '/data/djl/NeLoRa/OptimalPkt/'
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)
    file_paths = []
    for file_name in os.listdir(base_dir):
        if file_name.startswith('sf7') and file_name.endswith('.bin'):
            file_paths.append(os.path.join(base_dir, file_name))

    nsamp = round(n_classes * fs / bw)
    nfreq = 64 + 1
    time_upsamp = 4

    preamble_len = 8
    code_len = 2
    codes = [50, 101]  # TODO set codes
    fft_upsamp = 1024
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 2
    debug = True
    breakflag = True

    t = cp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    # if not gpu: t = t.get()
    # logger.debug(type(t))
    chirpI1 = mychirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    chirpQ1 = mychirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    upchirp = cp.array(chirpI1 + 1j * chirpQ1)

    chirpI1 = mychirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    chirpQ1 = mychirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    downchirp = cp.array(chirpI1 + 1j * chirpQ1)
    if use_gpu:
        plans = {1: fft.get_fft_plan(cp.zeros(nsamp * 1, dtype=cp.complex128)),
                 fft_upsamp: fft.get_fft_plan(cp.zeros(nsamp * fft_upsamp, dtype=cp.complex128))}
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
    global Config
    if not upsamp:
        upsamp = Config.fft_upsamp
    # upsamp = Config.fft_upsamp #!!!
    chirp_data = ndata * refchirp
    ans = cp.zeros(ndata.shape[0], dtype=cp.float64)
    phase = cp.zeros(ndata.shape[0], dtype=cp.complex128)
    for idx in range(ndata.shape[0]):
        fft_raw = myfft(chirp_data[idx], n=Config.nsamp * upsamp, plan=Config.plans[upsamp])
        target_nfft = Config.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(cp.float64) / upsamp
        phase[idx] = cut1[tocpu(cp.argmax(dat))]
        phase[idx] /= abs(phase[idx])
    return ans, phase


# noinspection SpellCheckingInspection
def dechirp(ndata, refchirp, upsamp=None):
    if len(ndata.shape) == 1:
        ndata = ndata.reshape(1, -1)
    global Config
    if not upsamp:
        upsamp = Config.fft_upsamp
    # upsamp = Config.fft_upsamp #!!!
    chirp_data = ndata * refchirp
    ans = cp.zeros(ndata.shape[0], dtype=cp.float64)
    power = cp.zeros(ndata.shape[0], dtype=cp.float64)
    for idx in range(ndata.shape[0]):
        fft_raw = myfft(chirp_data[idx], n=Config.nsamp * upsamp, plan=Config.plans[upsamp])
        target_nfft = Config.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(cp.float64) / upsamp
        power[idx] = cp.max(dat)
        # logger.debug(cp.argmax(dat), upsamp, ans[idx])
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

        fft_raw = cp.zeros((fft_n,))
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
    logger.info(
        f'coarse work: {argmax_est_time_shift_samples=}, {argmax_est_cfo_samples=}, {fft_n=}, {est_cfo_freq=} Hz, {est_to_s=} s)')
    return est_cfo_freq, argmax_est_time_shift_samples


def work(pkt_totcnt, pktdata_in):
    pktdata_in /= cp.mean(cp.abs(pktdata_in))
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
    debug_diff_0 = cp.unwrap(cp.array([x - cp.around(x) for x in ans1n]), discont=0.5)
    # sfo correction
    est_cfo_slope = all_cfo_freq / Config.sig_freq * Config.bw * Config.bw / Config.n_classes

    sig_time = len(pktdata3) / Config.fs
    logger.debug(f'{all_cfo_freq=} Hz, {est_cfo_slope=} Hz/s, {sig_time=} samples')
    t = cp.linspace(0, sig_time, len(pktdata3) + 1)[:-1]
    chirpI1 = mychirp(t, f0=0, f1=- est_cfo_slope * sig_time, t1=sig_time, method='linear', phi=90)
    chirpQ1 = mychirp(t, f0=0, f1=- est_cfo_slope * sig_time, t1=sig_time, method='linear', phi=0)
    est_cfo_symbol = cp.array(chirpI1 + 1j * chirpQ1)
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
        myscatter(range(len(angles)), angles, s=0.5)
        plt.savefig(os.path.join(Config.figpath, f'temp_sf7_{pkt_totcnt}.jpg'))
        plt.clf()
        for i in range(min(len(ans2n), len(angles))):
            if abs(angles[i]) > 0.5:
                logger.debug(f'abs(angles[i]) > 0.5: {i=} {cp_str(ans2n[i])} {cp_str(angles[i])}')

    return angles


def calc_angles(payload_data):
    angles = cp.zeros((len(payload_data),))
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
        logger.debug(f'decode: {pkt_totcnt=} power1 drops: {drop_idx=} {len(ans1n)=} {cp_str(power1)}')
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
    vals = cp.zeros((symb_cnt,), dtype=cp.float64)
    for i in range(symb_cnt - (Config.sfdpos + 2)):
        power = cp.sum(power1[i: i + Config.preamble_len]) + cp.sum(
            power2[i + Config.sfdpos: i + Config.sfdpos + 2])
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
    logger.debug(f'fine work {cp_str(ans1[: Config.preamble_len])} sfd {cp_str(ans2[Config.sfdpos: Config.sfdpos + 2])} {sfd_upcode=}, {sfd_downcode=}, {re_cfo_0=}, {est_to_0=}, {detect=}')
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


def gen_upchirp(t0, f0, t1, f1):
    t = cp.arange(math.ceil(t0), math.ceil(t1)) / Config.fs
    fslope = (f1 - f0) / (t1 - t0) * Config.fs
    chirpI1 = mychirp(t, f0=f0 - t0 / Config.fs * fslope, f1=f1, t1=t1 / Config.fs, method='linear', phi=90)
    chirpQ1 = mychirp(t, f0=f0 - t0 / Config.fs * fslope, f1=f1, t1=t1 / Config.fs, method='linear', phi=0)
    upchirp = cp.array(chirpI1 + 1j * chirpQ1)
    return upchirp


def fine_work_new(pktdata2a):  # TODO working
    est_cfo_freq = -26623.72589111328
    argmax_est_time_shift_samples = 534

    pktdata2a = togpu(pktdata2a)

    cfofreq_range = cp.linspace(-Config.bw / 4, Config.bw / 4, Config.nfreq)
    # detect_array_up = cp.zeros((Config.nfreq, Config.nsamp * Config.preamble_len * Config.time_upsamp), dtype=cp.float64)
    # detect_array_down = cp.zeros((Config.nfreq, Config.nsamp * 2 * Config.time_upsamp), dtype=cp.float64)

    est_cfo_percentile = est_cfo_freq / Config.sig_freq
    tsig = Config.nsamp / Config.fs * (1 - est_cfo_percentile)
    tstart_sig = 0
    dd = []
    fslope = Config.bw / tsig

    # test
    if 0:
        cfofreq = -26623.725
        time_error = 534
        pktdata2a_roll = cp.roll(pktdata2a, - time_error)
        detect_symb = gen_refchirp(cfofreq, 0)
        didx = 0
        res = 0
        for sidx, ssymb in enumerate(detect_symb[:8]):
            res += tocpu(cp.conj(togpu(ssymb)).dot(pktdata2a_roll[didx: didx + len(ssymb)]))
            # logger.info(f'{sidx=} {cp.abs(res)/len(ssymb)=} {cp.angle(res)=} {len(ssymb)=}')
            didx += len(ssymb)
        logger.info(f'try value {res=}')

    def objective(params):
        cfofreq, time_error = params
        pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
        detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error))
        didx = 0
        res = 0
        for sidx, ssymb in enumerate(detect_symb):
            ress = tocpu(cp.conj(togpu(ssymb)).dot(pktdata2a_roll[didx: didx + len(ssymb)]))
            # logger.debug(f'{sidx=} {abs(ress)=} {cp.angle(ress)=}')
            res += abs(ress) ** 2
            didx += len(ssymb)
        return -res  # Negative because we use a minimizer

    # Initial parameters
    initial_freq = -26623.725

    # Perform optimization
    if 0:
        bestx = None
        bestobj = cp.inf
        for start_t in tqdm(range(Config.nsamp)):
            for start_f in range(-27000, -26000, 100):
                result = opt.minimize(
                    objective,
                    [start_f + 50, start_t + 0.5],
                    bounds=[(start_f, start_f + 100), (start_t, start_t + 1)],
                    method='L-BFGS-B',  # More precise method for bounded problems
                    options={'gtol': 1e-8, 'disp': False}  # Set tolerance for convergence and display progress
                )
                logger.debug(f"Optimized parameters: cfofreq = {result.x[0]}, time_error = {result.x[1]} {result.fun=}")
                if result.fun < bestobj:
                    bestx = result.x
                    bestobj = result.fun
        cfo_freq_est, time_error = bestx
        logger.info(f"Optimized parameters: {cfo_freq_est=} {time_error=}")
    else:
        cfo_freq_est = -26695.219805083307
        time_error = 157.733830380595
    pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
    detect_symb, tstart_sig = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
    didx = 0
    for sidx, ssymb in enumerate(detect_symb[:8]):
        ress = tocpu(cp.conj(togpu(ssymb)).dot(pktdata2a_roll[didx: didx + len(ssymb)]))
        logger.info(f'{cp.angle(ress)=}')
        didx += len(ssymb)

    plt.clf()
    # phase = cp.angle(tocpu(pktdata2a_roll[:1024*20]))
    # unwrapped_phase = cp.unwrap(phase)
    # myplot(unwrapped_phase)
    # plt.axvline(tstart_sig, color="k")
    # plt.title("pkt20")
    # plt.savefig(os.path.join(Config.figpath,f"pk{code}.png"))
    # plt.clf()

    code_cnt = 101  # math.floor(len(pktdata2a_roll) / Config.nsamp - Config.sfdend - 0.5)
    code_ests = cp.zeros((code_cnt,), dtype=int)
    angle1 = cp.zeros((code_cnt,), dtype=float)
    angle2 = cp.zeros((code_cnt,), dtype=float)
    angle3 = cp.zeros((code_cnt,), dtype=float)
    angle4 = cp.zeros((code_cnt,), dtype=float)
    for codeid in range(code_cnt):
        logger.info(f"{tstart_sig=}")
        # tsig = Config.tsig * (1 - est_cfo_percentile)
        res_array = cp.zeros((Config.n_classes,), dtype=float)
        for code in range(Config.n_classes):
            tstart_sig1 = tstart_sig
            cfofreq = est_cfo_freq
            est_cfo_percentile = cfofreq / Config.sig_freq
            inif = code / Config.n_classes * Config.bw
            tsig = Config.tsig * (1 - est_cfo_percentile) * (1 - code / Config.n_classes)
            upchirp1 = gen_upchirp(tstart_sig1, -Config.bw / 2 + cfofreq + inif, tstart_sig1 + tsig, Config.bw / 2 + cfofreq)
            res1 = tocpu(cp.conj(togpu(upchirp1)).dot(pktdata2a_roll[math.ceil(tstart_sig1): math.ceil(tstart_sig1 + tsig)]))

            if code != 0:
                tstart_sig1 += tsig
                tsig = Config.tsig * (1 - est_cfo_percentile) * (code / Config.n_classes)
                upchirp2 = gen_upchirp(tstart_sig1, -Config.bw / 2 + cfofreq, tstart_sig1 + tsig, -Config.bw / 2 + cfofreq + inif)
                res2 = tocpu(cp.conj(togpu(upchirp2)).dot(pktdata2a_roll[math.ceil(tstart_sig1): math.ceil(tstart_sig1 + tsig)]))
                tstart_sig1 += tsig
            else:
                res2 = 0

            res = abs(res1) ** 2 + abs(res2) ** 2
            # logger.info(f"{code=} {abs(res1)=} {cp.angle(res1)=} {abs(res2)=} {cp.angle(res2)=}")
            res_array[code] = res
        est_code = cp.argmax(res_array)
        logger.info(f"{est_code=}")
        code_ests[codeid] = est_code
        myplot(res_array)
        plt.title("resarray")
        plt.axvline(est_code, color="k")
        plt.savefig(os.path.join(Config.figpath, f"pk{code}.png"))
        plt.clf()

        code = est_code
        tstart_sig1 = tstart_sig
        est_cfo_percentile = cfofreq / Config.sig_freq
        inif = code / Config.n_classes * Config.bw
        tsig = Config.tsig * (1 - est_cfo_percentile) * (1 - code / Config.n_classes)
        upchirp1 = gen_upchirp(tstart_sig1, -Config.bw / 2 + cfofreq + inif, tstart_sig1 + tsig, Config.bw / 2 + cfofreq)
        upchirp1 = cp.conj(upchirp1)
        res1 = tocpu(togpu(upchirp1).dot(pktdata2a_roll[math.ceil(tstart_sig1): math.ceil(tstart_sig1 + tsig)]))
        upchirp1 /= res1 / cp.abs(res1)
        angle1[codeid] = cp.angle(res1)
        logger.info(f"{abs(res1)=} {cp.angle(res1)=} {code=}")

        if code != 0:
            tstart_sig1 += tsig
            tsig = Config.tsig * (1 - est_cfo_percentile) * (code / Config.n_classes)
            upchirp2 = gen_upchirp(tstart_sig1, -Config.bw / 2 + cfofreq, tstart_sig1 + tsig, -Config.bw / 2 + cfofreq + inif)
            upchirp2 = cp.conj(upchirp2)
            res2 = tocpu(togpu(upchirp2).dot(pktdata2a_roll[math.ceil(tstart_sig1): math.ceil(tstart_sig1 + tsig)]))
            upchirp2 /= res2 / cp.abs(res2)
            angle2[codeid] = cp.angle(res2)
            tstart_sig1 += tsig
            logger.info(f"{abs(res2)=} {cp.angle(res2)=} {code=}")
        else:
            angle2[codeid] = 0

        dataX = pktdata2a_roll[math.ceil(tstart_sig): math.ceil(tstart_sig + Config.tsig * (1 - est_cfo_percentile))][:Config.nsamp]
        dataX = dataX.T
        data1 = cp.matmul(Config.dataE1, dataX)
        data2 = cp.matmul(Config.dataE2, dataX)
        vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
        est = tocpu(cp.argmax(vals))
        angle3[codeid] = tocpu(cp.angle(data1[est]))
        angle4[codeid] = tocpu(cp.angle(data2[est]))
        logger.info(f"{codeid=} {abs(tocpu(data1[est]))=} {abs(tocpu(data2[est]))=} {angle3[codeid]=} {angle4[codeid]=}")

        if code != 0:
            complex_array = cp.conj(cp.concatenate((upchirp1, upchirp2), axis=0))
        else:
            complex_array = upchirp1
            # Extract the phase and unwrap it
        phase = cp.angle(complex_array)
        unwrapped_phase = cp.unwrap(phase)

        # Compute the difference between consecutive unwrapped phase values
        phase_diff = cp.diff(unwrapped_phase)

        # Plotting
        plt.figure(figsize=(10, 6))

        myplot(unwrapped_phase, linestyle='-', color='b', label=f"{code=}")

        logger.info(f"{tstart_sig=} {tstart_sig1=}")
        complex_array = tocpu(pktdata2a_roll[math.ceil(tstart_sig): math.ceil(tstart_sig1)])
        # Extract the phase and unwrap it
        phase = cp.angle(complex_array)
        unwrapped_phase = cp.unwrap(phase)

        # Compute the difference between consecutive unwrapped phase values
        phase_diff = cp.diff(unwrapped_phase)

        # Plotting
        myplot(unwrapped_phase, linestyle='--', color='r', label="input")

        plt.title('Ref')
        plt.xlabel('Index')
        plt.ylabel('Phase')
        plt.grid(True)
        plt.axvline(Config.tsig * (1 - est_cfo_percentile) * (1 - code / Config.n_classes), color='k')
        plt.legend()
        plt.title(f'{codeid=}')
        plt.savefig(os.path.join(Config.figpath, f'code{codeid}.png'))
        plt.clf()

        tstart_sig = tstart_sig1
    myplot(angle1, label="angle1")
    myplot(angle2, label="angle2")
    plt.legend()
    plt.title(f"angleA.png")
    plt.savefig(os.path.join(Config.figpath, f"angleA.png"))
    plt.clf()
    myscatter(code_ests, angle1, label="angle1")
    myscatter(code_ests, angle2, label="angle2")
    plt.legend()
    plt.title(f"angleB.png")
    plt.savefig(os.path.join(Config.figpath, f"angleB.png"))
    plt.clf()

    myplot(angle3, label="angle3")
    myplot(angle4, label="angle4")
    plt.legend()
    plt.title(f"angleC.png")
    plt.savefig(os.path.join(Config.figpath, f"angleC.png"))
    plt.clf()
    myscatter(code_ests, angle3, label="angle3")
    myscatter(code_ests, angle4, label="angle4")
    plt.legend()
    plt.title(f"angleD.png")
    plt.savefig(os.path.join(Config.figpath, f"angleD.png"))
    plt.clf()

    sys.exit(0)
    tstart_range = cp.linspace(-1, 0, Config.time_upsamp + 1)[:-1]

    detect_array_up = [[[] for _ in tstart_range] for _ in cfofreq_range]
    progress_bar = tqdm(total=len(tstart_range) * len(cfofreq_range), desc="Generating Reference Chirps")
    for freq_idx, cfofreq in enumerate(cfofreq_range):
        for tstart_idx, tstart in enumerate(tstart_range):
            detect_symb = gen_refchirp(cfofreq, tstart)
            detect_array_up[freq_idx][tstart_idx] = togpu(cp.conj(cp.concatenate(detect_symb)))
            progress_bar.update(1)
    progress_bar.close()
    # logger.info(str([len(x) for x in detect_array_up]))
    # logger.info(f'{cfofreq=} {est_cfo_percentile=}')

    time_error_range = range(Config.nsamp)
    evals = cp.zeros((len(time_error_range), len(tstart_range), len(cfofreq_range)), dtype=float)
    if 0:
        progress_bar = tqdm(total=len(time_error_range) * len(tstart_range) * len(cfofreq_range), desc="Computing")
        # logger.info(f'{len(pktdata2a)=}')
        for time_error_idx, time_error in enumerate(time_error_range):
            for tstart_idx, tstart in enumerate(tstart_range):
                for freq_idx in range(len(cfofreq_range)):
                    pktdata2a_roll = cp.roll(pktdata2a, - time_error)
                    ssymb = detect_array_up[freq_idx][tstart_idx]
                    evals[time_error_idx][tstart_idx][freq_idx] = cp.abs(pktdata2a_roll[:len(ssymb)].dot(ssymb)) / len(ssymb)
                    progress_bar.update(1)
        progress_bar.close()

    max_evals = cp.unravel_index(cp.argmax(evals), evals.shape)
    max_evals = [int(x) for x in max_evals]
    time_error = time_error_range[max_evals[0]] + tstart_range[max_evals[1]]
    cfo_freq_est = cfofreq_range[max_evals[2]]

    logger.info(f'{max_evals=} {time_error=} samples, {cfo_freq_est=} Hz {cp.max(evals)=} \n\n')
    with open('eval.pkl', 'wb') as f:
        pickle.dump(evals, f)

    sys.exit(0)
    return time_error, cfo_freq_est


def gen_refchirp(cfofreq, tstart):
    est_cfo_percentile = cfofreq / Config.sig_freq
    detect_symb = []
    tstart_sig = tstart
    tsig = Config.tsig * (1 - est_cfo_percentile)
    for tid in range(Config.preamble_len):
        upchirp = gen_upchirp(tstart_sig, -Config.bw / 2 + cfofreq, tstart_sig + tsig, Config.bw / 2 + cfofreq)
        assert len(upchirp) == math.ceil(tstart_sig + tsig) - math.ceil(tstart_sig)
        upchirp[:20] = cp.zeros((20,))
        upchirp[-20:] = cp.zeros((20,))
        detect_symb.append(upchirp)
        tstart_sig += tsig
    # for tid in range(Config.code_len):
    #     inif = Config.codes[tid] / Config.nsamp * Config.bw
    #     tsig = Config.tsig * (1 - est_cfo_percentile) * (1 - Config.codes[tid] / Config.nsamp)
    #     upchirp = gen_upchirp(tstart_sig, -Config.bw / 2 + cfofreq + inif, tstart_sig + tsig, Config.bw / 2 + cfofreq)
    #     detect_symb.append(upchirp)
    #     tstart_sig += tsig
    #     tsig = Config.tsig * (1 - est_cfo_percentile) * (Config.codes[tid] / Config.nsamp)
    #     upchirp = gen_upchirp(tstart_sig, -Config.bw / 2 + cfofreq, tstart_sig + tsig, -Config.bw / 2 + cfofreq + inif)
    #     detect_symb.append(upchirp)
    #     tstart_sig += tsig
    detect_symb.append(cp.zeros((math.ceil(tstart_sig + tsig * 2) - math.ceil(tstart_sig),)))
    tstart_sig += tsig * 2
    for tid in range(2):
        upchirp = gen_upchirp(tstart_sig, Config.bw / 2 + cfofreq, tstart_sig + tsig, -Config.bw / 2 + cfofreq)  # TODO ???
        upchirp[:20] = cp.zeros((20,))
        upchirp[-20:] = cp.zeros((20,))
        detect_symb.append(upchirp)
        # logger.info(f'{len(upchirp) =} {tstart_sig + tsig=} {tstart_sig=}')
        assert len(upchirp) == math.ceil(tstart_sig + tsig) - math.ceil(tstart_sig)
        # logger.warning(f'warn {len(upchirp) =} {tstart_sig + tsig=} {tstart_sig=}')
        tstart_sig += tsig

    tsig = Config.tsig * (1 - est_cfo_percentile) * 0.25
    upchirp = gen_upchirp(tstart_sig, Config.bw / 2 + cfofreq, tstart_sig + tsig, Config.bw / 4 + cfofreq)  # TODO +-
    tstart_sig += tsig
    upchirp[:20] = cp.zeros((20,))
    upchirp[-20:] = cp.zeros((20,))
    assert len(upchirp) == math.ceil(tstart_sig + tsig) - math.ceil(tstart_sig)
    detect_symb.append(upchirp)
    return detect_symb, tstart_sig


# read packets from file
if __name__ == "__main__":
    angles = []
    plt.rcParams['font.size'] = 15
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(8, 6))

    for file_path in Config.file_paths:

        pkt_cnt = 0
        pktdata = []
        fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
        logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')

        power_eval_len = 5000
        nmaxs = cp.zeros((power_eval_len,))
        with open(file_path, "rb") as f:
            for i in tqdm(range(power_eval_len), disable=not Config.debug):  # while True:
                try:
                    rawdata = cp.fromfile(f, dtype=cp.complex64, count=Config.nsamp)
                except EOFError:
                    logger.debug("file complete")
                    break
                if len(rawdata) < Config.nsamp:
                    logger.debug("file complete", len(rawdata))
                    break
                nmaxs[i] = cp.max(cp.abs(rawdata))
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(tocpu(nmaxs.reshape(-1, 1)))
        thresh = cp.mean(kmeans.cluster_centers_)
        counts, bins = cp.histogram(nmaxs, bins=100)
        logger.debug(
            f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")

        pkt_totcnt = 0

        with open(file_path, "rb") as f:
            while True:
                try:
                    rawdata = cp.fromfile(f, dtype=cp.complex64, count=Config.nsamp)
                except EOFError:
                    logger.info("file complete with EOF")
                    break
                if len(rawdata) < Config.nsamp:
                    logger.debug(f"file complete, {len(rawdata)=}")
                    break
                nmax = cp.max(cp.abs(rawdata))
                # logger.debug(nmax)

                if nmax < thresh:
                    if len(pktdata) > 14 and pkt_cnt > 20:
                        if pkt_totcnt < 21:
                            # and False:
                            pktdata = []
                            pkt_totcnt += 1
                            continue
                        logger.info(f"start parsing pkt {pkt_totcnt} len: {len(pktdata)}")
                        # idata = cp.concatenate(pktdata)
                        # with open('test.dat', 'wb') as f: idata.tofile(f)
                        # sys.exit(0)

                        work(pkt_totcnt, cp.concatenate(pktdata))
                        if Config.breakflag:
                            logger.error("terminate after one packet")
                            sys.exit(0)
                        pkt_totcnt += 1
                    pkt_cnt += 1
                    pktdata = []
                else:
                    pktdata.append(rawdata)
