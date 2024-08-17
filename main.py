import logging
import os
import random
import sys
import time
import seaborn as sns
import cmath
import math
import matplotlib.pyplot as plt
# Enable fallback mode
# from cupyx.fallback_mode import numpy as np
import numpy as np
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.signal import chirp
from sklearn.cluster import KMeans
from tqdm import tqdm

use_gpu = True
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
    # logger.info(f"{f0=} {f1=} {t1=}")
    return togpu(chirp(tocpu(t), f0, t1, f1, method, phi))


def cp_str(x, precision=2, suppress_small=False):
    return np.array2string(tocpu(x), precision=precision, formatter={'float_kind': lambda k: f"{k:.2f}"}, floatmode='fixed', suppress_small=suppress_small)


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
# formatter = logging.Formatter('%(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(levelname)s - %(message)s')
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
    # base_dir = '/data/djl/datasets/Dataset_50Nodes'
    figpath = "fig"
    if not os.path.exists(figpath): os.mkdir(figpath)
    file_paths = ['/data/djl/datasets/Dataset_50Nodes/sf7-470-new-70.bin']
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
    chirpI1 = mychirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    chirpQ1 = mychirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    upchirp = cp.array(chirpI1 + 1j * chirpQ1)

    chirpI1 = mychirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    chirpQ1 = mychirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    downchirp = cp.array(chirpI1 + 1j * chirpQ1)
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
        plt.show()
        plt.savefig(os.path.join(Config.figpath, f'temp_sf7_{pkt_totcnt}.png'))
        plt.clf()
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


def gen_upchirp(t0, f0, t1, f1):
    t = cp.arange(math.ceil(t0), math.ceil(t1), dtype=float) / Config.fs
    fslope = (f1 - f0) / (t1 - t0) * Config.fs
    # logger.info(f"{fslope=} ref={Config.bw / Config.tsig} {f1=} {f0=} {t1=} {t0=} {t[0]} {t[-1]} {f0 - t0 / Config.fs * fslope}")
    chirpI1 = mychirp(t, f0=f0 - t0 / Config.fs * fslope, f1=f1, t1=t1 / Config.fs, method='linear', phi=90)
    chirpQ1 = mychirp(t, f0=f0 - t0 / Config.fs * fslope, f1=f1, t1=t1 / Config.fs, method='linear', phi=0)
    upchirp = cp.array(chirpI1 + 1j * chirpQ1, dtype=cp.complex64)
    # plt.figure(figsize=(20,3))
    # myplot(upchirp.real, marker='o')
    # plt.show()
    # sys.exit(1)
    return upchirp


def fine_work_new(pktdata2a):  # TODO working
    pktdata2a = togpu(pktdata2a)

    phase = cp.angle(pktdata2a)
    unwrapped_phase = cp.unwrap(phase)
    myplot(unwrapped_phase[:15 * 1024], linestyle='-')
    plt.title("input data 15 symbol")
    plt.show()
    plt.clf()

    # Perform optimization
    if 0:
        def objective(params):
            cfofreq, time_error = params
            pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
            detect_symb = gen_refchirp(cfofreq, time_error - math.ceil(time_error))
            didx = 0
            res = 0
            for sidx, ssymb in enumerate(detect_symb):
                ress = tocpu(cp.conj(ssymb).dot(pktdata2a_roll[didx: didx + len(ssymb)]))
                # logger.debug(f'{sidx=} {abs(ress)=} {cp.angle(ress)=}')
                res += abs(ress) ** 2
                didx += len(ssymb)
            return -res  # Negative because we use a minimizer

        bestx = None
        bestobj = cp.inf
        for tryidx in range(10000):

            # start_t = random.uniform(0, Config.nsamp)
            # start_f = random.uniform(-29000, -24000)
            ef = -24701.029772968563 - 2000
            et = 241.52649543073412
            start_t = random.uniform(et - 10, et + 10)
            start_f = random.uniform(ef - 500, ef + 500)
            result = opt.minimize(objective, [start_f, start_t], bounds=[(ef - 500, ef + 500), (et - 10, et + 10)], method='L-BFGS-B',  # More precise method for bounded problems
                                  options={'gtol': 1e-8, 'disp': False}  # Set tolerance for convergence and display progress
                                  )
            if result.fun < bestobj:
                logger.debug(f"{tryidx=: 6d} {result.fun=:.3f} cfofreq = {result.x[0]:.3f}, time_error = {result.x[1]:.3f} ")
                bestx = result.x
                bestobj = result.fun
        cfo_freq_est, time_error = bestx
        logger.info(f"Optimized parameters: {cfo_freq_est=} {time_error=}")

        if 0:
            start_t = np.linspace(0, Config.nsamp, Config.nsamp * 5)
            start_f = np.linspace(-24000, -29000, 100)
            Z = np.zeros((len(start_f), len(start_t)))
            for i, f in tqdm(enumerate(start_f), total=len(start_f)):
                for j, t in enumerate(start_t):
                    Z[i, j] = objective((f, t))
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(Z, cmap='viridis', cbar=True)
            ax.set_xticks(np.linspace(0, len(start_t) - 1, 10))
            ax.set_xticklabels(np.linspace(start_t[0], start_t[-1], 10))
            ax.set_yticks(np.linspace(0, len(start_f) - 1, 10))
            ax.set_yticklabels(np.linspace(start_f[0], start_f[-1], 10))
            ax.yaxis.set_tick_params(labelrotation=0)
            plt.xlabel('start_t')
            plt.ylabel('start_f')
            plt.title('Heatmap of objective(start_t, start_f)')
            plt.show()
            maxidx = np.unravel_index(np.argmin(Z, axis=None), Z.shape, order='C')
            best_f = start_t[maxidx[0]]
            best_t = start_t[maxidx[1]]
            logger.info(f'{objective((best_f, best_t))=} {np.min(Z)=} {best_f=} {best_t=}')
            sys.exit(0)
        cfo_freq_est_delta = 0  # 100
        time_error_delta = 0
    else:
        cfo_freq_est = -24701.029772968563
        time_error = 241.52649543073412
        cfo_freq_est_delta = -2000  # 100
        time_error_delta = 0
        cfo_freq_est += cfo_freq_est_delta
        time_error += time_error_delta

    pktdata2a_roll = cp.roll(pktdata2a, -math.ceil(time_error))
    phase1 = cp.angle(pktdata2a_roll)
    detect_symb = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
    detect_symb = cp.concatenate(detect_symb)
    detect_symb *= (pktdata2a_roll[0] / cp.abs(pktdata2a_roll[0]))
    tstart = time_error - math.ceil(time_error) + (Config.sfdend - 0.75) * Config.tsig * (1 - cfo_freq_est / Config.sig_freq)
    # xval = cp.arange(800, 1024)
    logger.info(len(detect_symb))
    xval = cp.arange(len(detect_symb))
    myplot(cp.unwrap(phase1)[xval], linestyle='-', color='b', label="input")
    myplot(cp.unwrap(cp.angle(detect_symb))[xval], linestyle='--', color='r', label="fit")
    # myplot(cp.diff(cp.unwrap(phase1))[xval], linestyle='-', color='b', label="input")
    # myplot(cp.diff(cp.unwrap(cp.angle(detect_symb)))[xval], linestyle='--', color='r', label="fit")
    plt.title("aligned pkt")
    plt.legend()
    plt.show()
    plt.clf()
    sys.exit(0)

    detect_symb = gen_refchirp(cfo_freq_est, time_error - math.ceil(time_error))
    didx = 0
    res_angle = cp.zeros((Config.preamble_len,), dtype=float)
    for sidx, ssymb in enumerate(detect_symb[:Config.preamble_len]):
        ress = cp.conj(togpu(ssymb)).dot(pktdata2a_roll[didx: didx + len(ssymb)])
        rangle = cp.angle(ress)
        res_angle[sidx] = rangle
        logger.info(f'{rangle=}')
        didx += len(ssymb)

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    # res_angle = res_angle[1:-1]  # TODO
    x_data = np.arange(len(res_angle), dtype=float)
    # add_data = cp.array([2,2,1,1,0,0,0,0])
    # res_angle += add_data * cp.pi * 2
    res_angle = np.unwrap(res_angle)
    # res_angle[-1] += np.pi * 2 # TODO
    params, covariance = curve_fit(quadratic, x_data, tocpu(res_angle))
    myscatter(x_data, res_angle, label='Data Points')
    myplot(x_data, quadratic(x_data, *params), color='red', label='Fitted Curve')
    # logger.info(f"fd={cfo_freq_est_delta} td={time_error_delta} a={params[0]} b={params[1]} c={params[2]}")
    plt.legend()
    plt.show()
    plt.savefig(f"res_angle.png")
    plt.clf()
    # sys.exit(0)

    code_cnt = math.floor(len(pktdata2a_roll) / Config.nsamp - Config.sfdend - 0.5)
    code_ests = cp.zeros((code_cnt,), dtype=int)
    angle1 = cp.zeros((code_cnt,), dtype=float)
    angle2 = cp.zeros((code_cnt,), dtype=float)
    angle3 = cp.zeros((code_cnt,), dtype=float)
    angle4 = cp.zeros((code_cnt,), dtype=float)
    tid_times = (cp.arange(code_cnt + 1, dtype=float) + Config.sfdpos + 2.25) * Config.tsig * (1 - cfo_freq_est / Config.sig_freq) + tstart
    inif_arr = cp.arange(Config.n_classes, dtype=float) / Config.n_classes * Config.bw
    tsig_arr = Config.tsig * (1 - cfo_freq_est / Config.sig_freq) * (1 - cp.arange(Config.n_classes, dtype=float) / Config.n_classes)
    for tid in range(code_cnt):

        upchirp1_arr = [gen_upchirp(tid_times[tid],
                                    -Config.bw / 2 + cfo_freq_est + inif_arr[code],
                                    tid_times[tid] + tsig_arr[code],
                                    Config.bw / 2 + cfo_freq_est)
                        for code in range(Config.n_classes)]
        upchirp2_arr = [gen_upchirp(tid_times[tid] + tsig_arr[code],
                                    -Config.bw / 2 + cfo_freq_est,
                                    tid_times[tid + 1],
                                    -Config.bw / 2 + cfo_freq_est + inif_arr[code]) if code != 0 else None
                        for code in range(Config.n_classes)]
        res1_arr = cp.zeros(Config.n_classes, dtype=cp.complex64)
        res2_arr = cp.zeros(Config.n_classes, dtype=cp.complex64)
        for code in range(Config.n_classes):
            res1_arr[code] = cp.conj(upchirp1_arr[code]).dot(pktdata2a_roll[math.ceil(tid_times[tid]): math.ceil(tid_times[tid] + tsig_arr[code])])
        for code in range(1, Config.n_classes):
            res2_arr[code] = cp.conj(upchirp2_arr[code]).dot(pktdata2a_roll[math.ceil(tid_times[tid] + tsig_arr[code]): math.ceil(tid_times[tid + 1])])
        res_array = cp.abs(res1_arr) ** 2 + cp.abs(res2_arr) ** 2
        est_code = tocpu(cp.argmax(res_array))
        logger.info(f"{est_code=} {cp.max(res_array)=}")
        code_ests[tid] = est_code
        myplot(res_array)
        plt.title(f"resarray {tid=} {est_code=}")
        plt.axvline(tocpu(est_code), color="k")
        plt.show()
        plt.savefig(os.path.join(Config.figpath, f"resarray {tid=} {est_code=}.png"))
        plt.clf()

        phase = cp.angle(pktdata2a_roll[math.ceil(tid_times[tid]): math.ceil(tid_times[tid + 1])])
        unwrapped_phase = cp.unwrap(phase)
        myplot(unwrapped_phase, linestyle='-')
        plt.title(f"phase {tid=} {est_code=}")
        plt.show()
        plt.clf()

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
        phase1 = cp.angle(upchirp_est)
        sigtt = cp.arange(math.ceil(tid_times[tid]), math.ceil(tid_times[tid + 1]), dtype=int)
        # plt.figure(figsize=(10, 6))
        myplot(sigtt, cp.unwrap(phase1), linestyle='-', color='b', label=f"{est_code=}")
        phase2 = cp.angle(pktdata2a_roll[sigtt])
        myplot(sigtt, cp.unwrap(phase2), linestyle='--', color='r', label="input")

        plt.title('Ref')
        plt.xlabel('Index')
        plt.ylabel('Phase')
        plt.grid(True)
        plt.axvline(tocpu(tid_times[tid] + tsig_arr[est_code]), color='k')
        plt.legend()
        plt.title(f'{tid=} {est_code=}')
        plt.show()
        plt.savefig(os.path.join(Config.figpath, f'code{tid}.png'))
        plt.clf()

        sys.exit(0)
    myplot(angle1, label="angle1")
    myplot(angle2, label="angle2")
    plt.legend()
    plt.title(f"angleA.png")
    plt.show()
    plt.savefig(os.path.join(Config.figpath, f"angleA.png"))
    plt.clf()
    myscatter(code_ests, angle1, label="angle1")
    myscatter(code_ests, angle2, label="angle2")
    plt.legend()
    plt.title(f"angleB.png")
    plt.show()
    plt.savefig(os.path.join(Config.figpath, f"angleB.png"))
    plt.clf()

    myplot(angle3, label="angle3")
    myplot(angle4, label="angle4")
    plt.legend()
    plt.title(f"angleC.png")
    plt.show()
    plt.savefig(os.path.join(Config.figpath, f"angleC.png"))
    plt.clf()
    myscatter(code_ests, angle3, label="angle3")
    myscatter(code_ests, angle4, label="angle4")
    plt.legend()
    plt.title(f"angleD.png")
    plt.show()
    plt.savefig(os.path.join(Config.figpath, f"angleD.png"))
    plt.clf()

    sys.exit(0)

    return time_error, cfo_freq_est


def gen_refchirp(cfofreq, tstart):
    est_cfo_percentile = cfofreq / Config.sig_freq
    detect_symb = []
    tind_times = cp.arange(Config.sfdend + 1, dtype=float)
    tind_times[-1] -= 0.75
    tid_times = tind_times * Config.tsig * (1 - est_cfo_percentile) + tstart
    # logger.info(est_cfo_percentile)
    for tid in range(Config.preamble_len):
        upchirp = gen_upchirp(tid_times[tid], -Config.bw / 2 + cfofreq, tid_times[tid + 1], Config.bw / 2 + cfofreq)
        assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        # upchirp[:20] = cp.zeros(20,dtype=cp.complex64)
        # upchirp[-20:] = cp.zeros(20,dtype=cp.complex64)
        detect_symb.append(upchirp)
        logger.warning(f"{tid=} {len(cp.concatenate(detect_symb))=}")
    for tid in range(Config.preamble_len, Config.sfdpos):
        detect_symb.append(cp.zeros(math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid]), dtype=cp.complex64))
        logger.warning(f"{tid=} {len(cp.concatenate(detect_symb))=}")
    for tid in range(Config.sfdpos, Config.sfdend):
        endfreq = - Config.bw / 2 if tid != Config.sfdend - 1 else Config.bw / 4
        upchirp = gen_upchirp(tid_times[tid], Config.bw / 2 + cfofreq, tid_times[tid + 1], endfreq + cfofreq)
        assert len(upchirp) == math.ceil(tid_times[tid + 1]) - math.ceil(tid_times[tid])
        logger.warning(f"{tid=} {Config.sfdend=} {len(cp.concatenate(detect_symb))=}")
        # upchirp[:20] = cp.zeros(20,dtype=cp.complex64)
        # upchirp[-20:] = cp.zeros(20,dtype=cp.complex64)
        detect_symb.append(upchirp)
    logger.warning(f"final {len(cp.concatenate(detect_symb))=}")
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
        nmaxs = cp.zeros(power_eval_len,dtype=float)
        for idx, rawdata in enumerate(read_large_file(file_path)):
            nmaxs[idx] = cp.max(cp.abs(rawdata))
            if idx == power_eval_len - 1: break
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(tocpu(nmaxs.reshape(-1, 1)))
        thresh = cp.mean(kmeans.cluster_centers_)
        counts, bins = cp.histogram(nmaxs, bins=100)
        logger.info(f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")

        pkt_totcnt = 0
        for pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=20)):
            if pkt_idx < 20: continue
            logger.info(f"{pkt_idx=} {len(pkt_data)=}")
            work(pkt_idx, pkt_data)
