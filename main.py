import cmath
import colorsys
import math
import os
import pickle

import cupy as cp
import cupyx.scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
from sklearn.cluster import KMeans

from tqdm import tqdm
import sys

import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)  # Set the logger level to debug
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the console handler level
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(logging.INFO)  # Set the file handler level
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class ModulusComputation:
    @staticmethod
    def average_modulus(lst, n_classes):
        complex_sum = cp.sum(cp.exp(1j * 2 * cp.pi * cp.array(lst) / n_classes))
        avg_angle = cp.angle(complex_sum)
        avg_modulus = (avg_angle / (2 * cp.pi)) * n_classes
        return avg_modulus


class Config:
    sf = 7
    bw = 125e3
    fs = 1e6
    sig_freq = 470e6
    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs # in samples
    base_dir = '/data/djl/NeLoRa/OptimalPkt/'

    file_paths = []
    for file_name in os.listdir(base_dir):
        if file_name.startswith('sf7') and file_name.endswith('.bin'):
            file_paths.append(os.path.join(base_dir, file_name))

    nsamp = round(n_classes * fs / bw)
    nfreq = 256 + 1
    time_upsamp = 4

    preamble_len = 8
    code_len = 2
    codes = [50, 101]  # TODO set codes
    fft_upsamp = 1024
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 2
    debug = True
    breakflag = True

    t = np.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    chirpI1 = chirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    chirpQ1 = chirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    upchirp = cp.array(chirpI1 + 1j * chirpQ1)

    chirpI1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
    chirpQ1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
    downchirp = cp.array(chirpI1 + 1j * chirpQ1)
    plans = {1: fft.get_fft_plan(cp.zeros(nsamp * 1, dtype=cp.complex128)),
             fft_upsamp: fft.get_fft_plan(cp.zeros(nsamp * fft_upsamp, dtype=cp.complex128))}

    dataE1 = cp.zeros((n_classes, nsamp), dtype=cp.cfloat)
    dataE2 = cp.zeros((n_classes, nsamp), dtype=cp.cfloat)
    for symbol_index in range(n_classes):
        time_shift = int(symbol_index / n_classes * nsamp)
        time_split = nsamp - time_shift
        dataE1[symbol_index][:time_split] = downchirp[time_shift:]
        if symbol_index != 0: dataE2[symbol_index][time_split:] = downchirp[:time_shift]


cp.cuda.Device(0).use()
opts = Config()
Config = Config()


def dechirp_phase(ndata, refchirp, upsamp=None):
    if len(ndata.shape) == 1:
        ndata = ndata.reshape(1, -1)
    global Config
    if not upsamp:
        upsamp = Config.fft_upsamp
    # upsamp = Config.fft_upsamp #!!!
    chirp_data = ndata * refchirp
    ans = cp.zeros(ndata.shape[0], dtype=cp.float64)
    phase = cp.zeros(ndata.shape[0], dtype=cp.float64)
    for idx in range(ndata.shape[0]):
        fft_raw = fft.fft(chirp_data[idx], n=Config.nsamp * upsamp, plan=Config.plans[upsamp])
        target_nfft = Config.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(cp.float64) / upsamp
        phase[idx] = cut1[int(cp.argmax(dat))]
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
        fft_raw = fft.fft(chirp_data[idx], n=Config.nsamp * upsamp, plan=Config.plans[upsamp])
        target_nfft = Config.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(cp.float64) / upsamp
        power[idx] = cp.max(dat)
        # logger.debug(cp.argmax(dat), upsamp, ans[idx])
    return ans, power


def add_freq(pktdata_in, est_cfo_freq):
    cfosymb = cp.exp(2j * np.pi * est_cfo_freq * cp.linspace(0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
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
            fft_raw_1 = fft.fft(sig1, n=fft_n, plan=Config.plans[Config.fft_upsamp])
            fft_raw += cp.abs(fft_raw_1) ** 2
        for sfd_idx in range(Config.sfdpos, Config.sfdpos + 2):
            sig2_pos = est_time_shift_samples + Config.nsamp * sfd_idx
            sig2 = pktdata_in[sig2_pos: sig2_pos + Config.nsamp] * Config.upchirp
            fft_raw_2 = fft.fft(sig2, n=fft_n, plan=Config.plans[Config.fft_upsamp])
            fft_raw += cp.abs(fft_raw_2) ** 2
        max_val = cp.max(fft_raw)
        if max_val > argmax_val:
            argmax_val = max_val
            argmax_est_time_shift_samples = est_time_shift_samples
            argmax_est_cfo_samples = cp.argmax(fft_raw)
    if argmax_est_cfo_samples > fft_n / 2:
        argmax_est_cfo_samples -= fft_n
    est_cfo_freq = argmax_est_cfo_samples.get() * (Config.fs / fft_n)
    est_to_s = argmax_est_time_shift_samples / Config.fs
    logger.info(
        f'coarse work: {argmax_est_time_shift_samples=}, {argmax_est_cfo_samples=}, {fft_n=}, {est_cfo_freq=} Hz, {est_to_s=} s)')
    return est_cfo_freq, argmax_est_time_shift_samples


def work(pkt_totcnt, pktdata_in):
    fft_n = Config.nsamp * Config.fft_upsamp
    if False:  # !!! TODO  False
        est_cfo_freq, argmax_est_time_shift_samples = coarse_work(pktdata_in)
        pktdata2a = add_freq(pktdata_in, - est_cfo_freq)
        pktdata2a = np.roll(pktdata2a, -argmax_est_time_shift_samples)

        # second detection
        # ====
        est_to_dec, est_to_int, pktdata3, re_cfo_0, re_cfo_freq, detect = fine_work(pktdata2a)

        # argmax_est_time_shift_samples = 534
        # argmax_est_cfo_samples = -27917
        # fft_n = 1048576
        # est_cfo_freq = -26623.72589111328
        #
        # est_to_s = 0.000534
        # re_cfo_0 = -0.0021972827444545146
        # est_to_int = 0
        # est_to_dec = 0.017822248505544945
        # detect = 0
        with open("temp.pkl", "wb") as f:
            pickle.dump((est_to_dec, est_to_int, pktdata3, re_cfo_0, re_cfo_freq, detect, est_cfo_freq,
                         argmax_est_time_shift_samples, pktdata2a), f)
    with open("temp.pkl", "rb") as f:
        est_to_dec, est_to_int, pktdata3, re_cfo_0, re_cfo_freq, detect, est_cfo_freq, argmax_est_time_shift_samples, pktdata2a = pickle.load(
            f)

    logger.info(
        f"fine work old {est_cfo_freq=}, {argmax_est_time_shift_samples=}, {re_cfo_0=}, {est_to_int=}, {est_to_dec=} {detect=}")
    # fine_work_new(pktdata_in)

    # if Config.breakflag: sys.exit(0)
    all_cfo_freq = re_cfo_freq + est_cfo_freq

    detect, upsamp = test_preamble(est_to_dec, pktdata3)

    pktdata4 = pktdata3[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]

    ans1n, ndatas = decode_payload(detect, est_to_dec, pktdata4, pkt_totcnt)

    logger.debug('decoded data ' + ' '.join([f'{x:.3f}' for x in ans1n]))
    # debug_diff_0 = np.unwrap([x - round(x) for x in ans1n], discont=0.5)
    # sfo correction
    est_cfo_slope = all_cfo_freq / Config.sig_freq * Config.bw * Config.bw / Config.n_classes

    sig_time = len(pktdata3) / Config.fs
    logger.info(f'{all_cfo_freq=}Hz, {est_cfo_slope=}Hz/s, {sig_time=}')
    t = np.linspace(0, sig_time, len(pktdata3) + 1)[:-1]
    chirpI1 = chirp(t, f0=0, f1=- est_cfo_slope * sig_time, t1=sig_time, method='linear', phi=90)
    chirpQ1 = chirp(t, f0=0, f1=- est_cfo_slope * sig_time, t1=sig_time, method='linear', phi=0)
    est_cfo_symbol = cp.array(chirpI1 + 1j * chirpQ1)
    pktdata5 = pktdata3 * est_cfo_symbol
    detect, upsamp = test_preamble(est_to_dec, pktdata5)

    est_to_dec2, est_to_int2, pktdata6, re_cfo_2, re_cfo_freq_2, detect = fine_work(pktdata5)
    all_cfo_freq = re_cfo_freq + est_cfo_freq + re_cfo_freq_2
    detect, upsamp = test_preamble(est_to_dec2, pktdata6)
    pktdata7 = pktdata6[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]
    ans2n, ndatas = decode_payload(detect, est_to_dec, pktdata7, pkt_totcnt)

    logger.info(f'decoded data {len(ans2n)=} ' + ' '.join([f'{x:.3f}' for x in ans2n]))


def decode_payload(detect, est_to_dec, pktdata4, pkt_totcnt):
    symb_cnt = len(pktdata4) // Config.nsamp
    ndatas = pktdata4[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)
    ans1, power1 = dechirp(ndatas[detect + Config.preamble_len: detect + Config.preamble_len + 2], Config.downchirp)
    ans2, power2 = dechirp(ndatas[detect + Config.sfdpos + 2:], Config.downchirp)
    ans1 = np.concatenate((ans1, ans2), axis=0)
    power1 = np.concatenate((power1, power2), axis=0)
    ans1n = ans1.get()
    ans1n += est_to_dec / 8
    ans1r = [round(x) % Config.n_classes for x in ans1n]
    if not min(power1) > np.mean(power1) / 2:
        drop_idx = next((idx for idx, num in enumerate(power1) if num < np.mean(power1) / 2), -1)
        ans1n = ans1n[:drop_idx]
        logger.debug(
            f'decode: {pkt_totcnt=} power1 drops: {drop_idx=} {len(ans1n)=} {" ".join([str(round(x.item())) for x in power1])}')
    return ans1n, ndatas


def test_preamble(est_to_dec, pktdata3):
    ndatas2a = pktdata3[: Config.sfdend * Config.nsamp].reshape(Config.sfdend, Config.nsamp)

    upsamp = Config.fft_upsamp
    detect = 0
    ans1, power1 = dechirp(ndatas2a[detect: detect + Config.preamble_len], Config.downchirp, upsamp)
    ans1 = ans1.get()
    ans1 += est_to_dec / 8
    ans2, power2 = dechirp(ndatas2a[detect + Config.sfdpos: detect + Config.sfdpos + 2], Config.upchirp, upsamp)
    ans2 = ans2.get()
    ans2 += est_to_dec / 8
    logger.info(
        'preamble: ' + " ".join([f'{x:.3f}' for x in ans1]) + 'sfd: ' + " ".join([f'{x:.3f}' for x in ans2]) + str(
            est_to_dec / 8))
    logger.info('power' + np.array2string(power1, precision=2))

    ans3, phase = dechirp_phase(ndatas2a[detect: detect + Config.preamble_len], Config.downchirp, upsamp)
    logger.info('preamble: ' + " ".join([f'{x:.3f}' for x in ans3]))
    logger.info('phase: ' + " ".join([f'{cp.angle(x):.3f}' for x in phase]))
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
    ansval = ModulusComputation.average_modulus(ans1[detect: detect + Config.preamble_len], Config.n_classes)
    sfd_upcode = ansval.get()
    ansval2 = ModulusComputation.average_modulus(ans2[detect + Config.sfdpos: detect + Config.sfdpos + 2],
                                                 Config.n_classes)
    sfd_downcode = ansval2.get()
    re_cfo_0 = ModulusComputation.average_modulus((sfd_upcode, sfd_downcode), Config.n_classes)
    est_to_0 = ModulusComputation.average_modulus((sfd_upcode, - sfd_downcode), Config.n_classes)
    if opts.debug:
        logger.debug('fine work' + ' '.join([f'{x:.3f}' for x in ans1[: Config.preamble_len]]) + 'sfd' +
                     ' '.join([f'{x:.3f}' for x in ans2[Config.sfdpos: Config.sfdpos + 2]]) +
                     f'{sfd_upcode=}, {sfd_downcode=}, {re_cfo_0=}, {est_to_0=}, {detect=}')
    logger.debug('fine work angles: preamble')
    for sig in ndatas[detect: detect + Config.preamble_len]:
        chirp_data = sig * Config.downchirp
        upsamp = Config.fft_upsamp
        fft_raw = fft.fft(chirp_data, n=Config.nsamp * upsamp, plan=Config.plans[upsamp])
        target_nfft = Config.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans = round(cp.argmax(dat).get().item() / upsamp)

        logger.debug(cmath.phase(cut1[ans]), cmath.phase(cut2[ans]))

    est_to_0 = est_to_0.get().item()
    re_cfo_0 = re_cfo_0.get().item()
    re_cfo_freq = re_cfo_0 * (Config.fs / fft_n)
    est_to_int = round(est_to_0)
    est_to_dec = est_to_0 - est_to_int
    pktdata3 = add_freq(pktdata2a, - re_cfo_freq)
    pktdata3 = np.roll(pktdata3, -est_to_int)
    return est_to_dec, est_to_int, pktdata3, re_cfo_0, re_cfo_freq, detect


def gen_upchirp(t0, f0, t1, f1):
    t = np.arange(np.ceil(t0), np.ceil(t1))
    fslope = (f1 - f0) / (t1 - t0)
    chirpI1 = chirp(t, f0=f0 - t0 * fslope, f1=f1, t1=t1, method='linear', phi=90)
    chirpQ1 = chirp(t, f0=f0 - t0 * fslope, f1=f1, t1=t1, method='linear', phi=0)
    upchirp = cp.array(chirpI1 + 1j * chirpQ1)
    return upchirp



def fine_work_new(pktdata2a):  # TODO working
    est_cfo_freq = -26623.72589111328
    argmax_est_time_shift_samples = 534

    pktdata2a = cp.array(pktdata2a)
    cfofreq_range = np.linspace(-Config.bw / 4, Config.bw / 4, Config.nfreq)
    # detect_array_up = cp.zeros((Config.nfreq, Config.nsamp * Config.preamble_len * Config.time_upsamp), dtype=cp.float64)
    # detect_array_down = cp.zeros((Config.nfreq, Config.nsamp * 2 * Config.time_upsamp), dtype=cp.float64)

    est_cfo_percentile = est_cfo_freq / Config.sig_freq
    tsig = Config.nsamp / Config.fs * (1 - est_cfo_percentile)
    tstart_sig = 0
    dd = []
    fslope = Config.bw / tsig

    tstart_range = np.linspace(-1, 0, Config.time_upsamp + 1)[:-1]
    detect_array_up = [[[] for _ in cfofreq_range] for _ in tstart_range]
    for freq_idx, cfofreq in enumerate(cfofreq_range):
        est_cfo_percentile = cfofreq / Config.sig_freq
        for tstart_idx, tstart in enumerate(tstart_range):
            detect_symb = []
            tstart_sig = tstart
            for tid in range(Config.preamble_len):
                tsig = Config.tsig * (1 - est_cfo_percentile)
                upchirp = gen_upchirp(tstart_sig, -Config.bw / 2 + cfofreq, tstart_sig + tsig, Config.bw / 2 + cfofreq)
                detect_symb.append(upchirp)
                tstart_sig += tsig

            for tid in range(Config.code_len):
                inif = Config.codes[tid] / Config.nsamp * Config.bw
                tsig = Config.tsig * (1 - est_cfo_percentile) * (1 - Config.codes[tid] / Config.nsamp)
                upchirp = gen_upchirp(tstart_sig, -Config.bw / 2 + cfofreq + inif, tstart_sig + tsig, Config.bw / 2 + cfofreq)
                detect_symb.append(upchirp)
                tstart_sig += tsig
                tsig = Config.tsig * (1 - est_cfo_percentile) * (Config.codes[tid] / Config.nsamp)
                upchirp = gen_upchirp(tstart_sig, -Config.bw / 2 + cfofreq, tstart_sig + tsig, -Config.bw / 2 + cfofreq + inif)
                detect_symb.append(upchirp)
                tstart_sig += tsig

            for tid in range(2):
                tsig = Config.tsig * (1 - est_cfo_percentile)
                upchirp = gen_upchirp(tstart_sig, -Config.bw / 2 - cfofreq, tstart_sig + tsig, Config.bw / 2 - cfofreq)
                detect_symb.append(cp.conj(upchirp))
                tstart_sig += tsig
            tsig = Config.tsig * (1 - est_cfo_percentile) * 0.25
            upchirp = gen_upchirp(tstart_sig, -Config.bw / 2 - cfofreq, tstart_sig + tsig, -Config.bw / 4 - cfofreq)
            detect_symb.append(cp.conj(upchirp))
            tstart_sig += tsig

            detect_array_up[freq_idx][tstart_idx] = cp.conj(cp.concatenate(detect_symb, axis=0))

    # logger.info(str([len(x) for x in detect_array_up]))
    # logger.info(f'{cfofreq=} {est_cfo_percentile=}')

    time_error_range = range(Config.nsamp)
    evals = cp.zeros((len(time_error_range), len(tstart_range), len(cfofreq_range)), dtype=float)
    # logger.info(f'{len(pktdata2a)=}')
    for time_error_idx, time_error in enumerate(time_error_range):
        for tstart_idx, tstart in enumerate(tstart_range):
            for freq_idx in range(len(cfofreq_range)):
                pktdata2a_roll = cp.roll(pktdata2a, - time_error)
                arr = detect_array_up[freq_idx][tstart_idx]
                # logger.info(f'{time_error_idx=} {small_time_error=} {len(arr)=} {arr.shape=}')
                # logger.info(f'{len(pktdata2a_roll)=} {(pktdata2a_roll[:len(arr)]).shape=}')
                evals[time_error_idx][tstart_idx][freq_idx] = cp.abs(cp.dot(pktdata2a_roll[:len(arr)], arr)) / len(arr)

    max_evals = cp.unravel_index(cp.argmax(evals), evals.shape)
    max_evals = [int(x) for x in max_evals]
    time_error = time_error_range[max_evals[0]] + tstart_range[max_evals[1]]
    cfo_freq_est = cfofreq_range[max_evals[2]]

    logger.info(f'{max_evals=} {time_error=} samples, {cfo_freq_est=} Hz {cp.max(evals)=} \n\n')
    return time_error, cfo_freq_est


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
        nmaxs = np.zeros((power_eval_len,))
        with open(file_path, "rb") as f:
            for i in tqdm(range(power_eval_len), disable=not Config.debug):  # while True:
                try:
                    rawdata = np.fromfile(f, dtype=cp.complex64, count=Config.nsamp)
                except EOFError:
                    logger.debug("file complete")
                    break
                if len(rawdata) < Config.nsamp:
                    logger.debug("file complete", len(rawdata))
                    break
                nmaxs[i] = np.max(np.abs(rawdata))
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(nmaxs.reshape(-1, 1))
        thresh = np.mean(kmeans.cluster_centers_)
        if opts.debug:
            counts, bins = np.histogram(nmaxs, bins=100)
            logger.debug(
                f"Init file find cluster: counts={np.array2string(counts, precision=2, suppress_small=True)}, bins={np.array2string(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")

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
                        if pkt_totcnt < 20 and False:
                            pktdata = []
                            pkt_totcnt += 1
                            continue
                        logger.info(f"start parsing pkt {pkt_totcnt} len: {len(pktdata)}")
                        # idata = cp.concatenate(pktdata).get()
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
