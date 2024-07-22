import colorsys
import os
import pickle
import shutil
import cmath
import math
import sys

import cupy as cp
import cupyx.scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
from tqdm import tqdm


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
    n_classes = 2 ** sf
    symb_cnt = 148
    nsamp = round(n_classes * fs / bw)

    preamble_len = 8
    code_len = 2
    fft_upsamp = 128
    sfdpos = preamble_len + code_len
    debug = False

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
        # if Config.debug: print(cp.argmax(dat), upsamp, ans[idx])
    return ans, power


def work(pktdata_in):

    argmax_est_time_shift_samples = 0
    argmax_est_cfo_samples = 0
    argmax_val = 0
    fft_n = Config.nsamp * Config.fft_upsamp

    for est_time_shift_samples in range(Config.nsamp * 2):

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

    # print(f'{argmax_est_time_shift_samples=}, {argmax_est_cfo_samples=}, {fft_n=}, {est_cfo_freq=} Hz, {est_to_s=} s)')
    cfosymb = cp.exp(- 2j * np.pi * est_cfo_freq * cp.linspace(0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
    pktdata2a = pktdata_in * cfosymb
    est_to_int = round(est_to_s)
    est_to_dec = est_to_s - est_to_int
    pktdata2a = np.roll(pktdata2a, -argmax_est_time_shift_samples)  # !!! TODO est_to_dec
    symb_cnt = 15
    ndatas2a = pktdata2a[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)

    upsamp = Config.fft_upsamp
    detect = 0
    ans1, power1 = dechirp(ndatas2a[detect: detect + Config.preamble_len], Config.downchirp, upsamp)
    ans2, power2 = dechirp(ndatas2a[detect + Config.sfdpos: detect + Config.sfdpos + 2], Config.upchirp, upsamp)
    # print(f'preamble: {" ".join([str(round(x.item())) for x in ans1])}, sfd: {" ".join([str(round(x.item())) for x in ans2])}')

    pktdata2 = pktdata2a[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]

    ndatas = pktdata2[: Config.symb_cnt * Config.nsamp].reshape(Config.symb_cnt, Config.nsamp)
    ans1, power1 = dechirp(ndatas[detect + Config.sfdpos + 2:], Config.downchirp)
    ans1n = ans1.get()

    '''
    if Config.debug: print('est_to_dec / 8', est_to_dec / (Config.nsamp / Config.n_classes))
    ans1n += est_to_dec / (Config.nsamp / Config.n_classes)  # ???????????????''' # TODO !!!
    assert min(power1) > np.mean(power1) / 2, f'power1 drops: {" ".join([str(round(x.item())) for x in power1])}'
    print('decoded data', ' '.join([str(round(x.item())) for x in ans1n]))
    angles = []
    for dataY in ndatas[detect + Config.sfdpos + 4:]:
        opts = Config
        dataX = cp.array(dataY)
        data1 = cp.abs(cp.matmul(Config.dataE1, dataX))
        data2 = cp.abs(cp.matmul(Config.dataE2, dataX))
        vals = data1 + data2
        est = cp.argmax(vals)
        if est > 0:
            time_shift = int(est / opts.n_classes * opts.nsamp)
            time_split = opts.nsamp - time_shift
            data1 = dataY[:time_split] * opts.downchirp[time_shift:]
            data2 = dataY[time_split:] * opts.downchirp[:time_shift]
            avg1 = cmath.phase(np.sum(data1))
            avg2 = cmath.phase(np.sum(data2))
            diff_avg = avg2 - avg1
            if diff_avg < -math.pi:
                diff_avg += math.pi * 2
            angles.append(diff_avg)
    return angles


# read packets from file
if __name__ == "__main__":
    angles = []
    plt.rcParams['font.size'] = 15
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(8, 6))
    for file_path in os.listdir('.'):
        print(file_path)
        if not file_path.endswith('.bin'): continue
        Config.file_path = file_path
        thresh = 0.002
        pkt_cnt = 0
        pktdata = []
        fsize = int(os.stat(Config.file_path).st_size / (Config.nsamp * 4 * 2))
        if Config.debug: print(f'reading file: {Config.file_path} SF: {Config.sf} pkts in file: {fsize}')
        nmaxs = []
        with open(Config.file_path, "rb") as f:
            for i in range(5000):  # while True:
                try:
                    rawdata = np.fromfile(f, dtype=cp.complex64, count=Config.nsamp)
                except EOFError:
                    if Config.debug: print("file complete")
                    break
                if len(rawdata) < Config.nsamp:
                    if Config.debug: print("file complete", len(rawdata))
                    break
                nmaxs.append(np.max(np.abs(rawdata)))
        counts, bins = np.histogram(nmaxs, bins=100)
        if Config.debug: print(counts, bins)

        pkt_totcnt = 0

        with open(Config.file_path, "rb") as f:
            while True:
                try:
                    rawdata = cp.fromfile(f, dtype=cp.complex64, count=Config.nsamp)
                except EOFError:
                    if Config.debug: print("file complete")
                    break
                if len(rawdata) < Config.nsamp:
                    if Config.debug: print("file complete", len(rawdata))
                    break
                nmax = cp.max(cp.abs(rawdata))
                # if Config.debug: print(nmax)

                if nmax < thresh:
                    if len(pktdata) > 14 and pkt_cnt > 20:
                        if Config.debug: print(f"start parsing pkt {pkt_totcnt} len: {len(pktdata)}")
                        angles.extend(work(cp.concatenate(pktdata)))
                        color = [colorsys.hls_to_rgb((hue + 0.5) / 3, 0.4, 1) for hue in [0, 1, 2]]

                        name = 'angles'
                        # Draw Figs
                        data = angles
                        count, bins_count = np.histogram(data, range=(-np.pi, np.pi), bins=100)
                        pdf = count / sum(count)

                        cdf = np.cumsum(pdf)

                        plt.plot(bins_count[1:], cdf, label=name)
                        plt.xlim(-np.pi, np.pi)
                        plt.xlabel('Angle (rad)')
                        plt.ylabel('Frequency')
                        plt.legend()
                        plt.savefig(name + '.pdf')
                        plt.savefig(name + '.png')
                        with open(name + '.pkl', 'wb') as g:
                            pickle.dump(angles, g)
                        plt.clf()

                        pkt_totcnt += 1
                    pkt_cnt += 1
                    pktdata = []
                else:
                    pktdata.append(rawdata)
