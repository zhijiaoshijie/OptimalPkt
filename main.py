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

import sys

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
    symb_cnt = 148
    symb_truth = np.array([
        110, 43, 91, 78, 10, 109, 39, 109, 76, 103, 110, 10, 90, 52, 40, 80, 6, 12, 118, 68, 104, 79, 99, 59, 120, 23,
        84, 64, 123, 11, 108, 55, 20, 48, 69, 102, 25, 50, 29, 40, 96, 71, 115, 86, 4, 8, 113, 63, 125, 2, 1, 52, 4, 8,
        16, 32, 100, 58, 28, 62, 105, 48, 32, 32, 69, 122, 109, 23, 97, 63, 68, 71, 114, 22, 81, 90, 126, 6, 117, 10,
        43, 93, 116, 31, 56, 18, 29, 25, 39, 76, 85, 122, 62, 124, 55, 77, 111, 46, 95, 0, 7, 116, 26, 20, 57, 0, 3, 1,
        7, 13, 25, 49, 9, 17, 59, 27, 28, 56, 17, 64, 93, 79, 18, 89, 114, 29, 121, 47, 86, 91, 120, 32, 60, 10, 45, 121
    ])

    nsamp = round(n_classes * fs / bw)

    preamble_len = 8
    code_len = 2
    fft_upsamp = 4096
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 2
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
    pktdata2a = add_freq(pktdata_in, - est_cfo_freq)
    pktdata2a = np.roll(pktdata2a, -argmax_est_time_shift_samples)  # !!! TODO est_to_dec
    return est_cfo_freq, pktdata2a


def work(pkt_totcnt, pktdata_in):
    fft_n = Config.nsamp * Config.fft_upsamp
    est_cfo_freq, pktdata2a = coarse_work(pktdata_in)
    # second detection
    # ====
    est_to_dec, est_to_int, pktdata3, re_cfo_0, re_cfo_freq = fine_work(pktdata2a)
    # print(f"{re_cfo_0=}, {est_to_int=}, {est_to_dec=}")
    all_cfo_freq = re_cfo_freq + est_cfo_freq

    detect, upsamp = test_preamble(est_to_dec, pktdata3)

    pktdata4 = pktdata3[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]

    if len(pktdata4) < Config.symb_cnt * Config.nsamp:
        print(f'{pkt_totcnt=} short signal: {len(pktdata4)=} {len(pktdata4) / Config.nsamp=} < {Config.symb_cnt=}')
        return []
    ans1n, ndatas = decode_payload(detect, est_to_dec, pktdata4, pkt_totcnt)
    angles1= []

    for dataY in ndatas[detect + Config.sfdpos + 4:]:
        opts = Config
        dataX = cp.array(dataY)
        data1 = cp.matmul(Config.dataE1, dataX)
        data2 = cp.matmul(Config.dataE2, dataX)
        vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
        est = cp.argmax(vals)
        if est > 0:
            diff_avg0 = cmath.phase(data2[est] / data1[est])
            angles1.append(diff_avg0)


    angles0 = []
    angles = []
    # print('decoded data', ' '.join([f'{x:.3f}' for x in ans1n]))
    # sfo correction
    est_cfo_slope = all_cfo_freq / Config.sig_freq * Config.bw * Config.bw / Config.n_classes

    sig_time = len(pktdata3) / Config.fs
    # print(all_cfo_freq, 'Hz', est_cfo_slope, 'Hz/s', sig_time)
    t = np.linspace(0, sig_time, len(pktdata3) + 1)[:-1]
    chirpI1 = chirp(t, f0=0, f1=- est_cfo_slope * sig_time, t1=sig_time, method='linear', phi=90)
    chirpQ1 = chirp(t, f0=0, f1=- est_cfo_slope * sig_time, t1=sig_time, method='linear', phi=0)
    est_cfo_symbol = cp.array(chirpI1 + 1j * chirpQ1)
    pktdata5 = pktdata3 * est_cfo_symbol
    detect, upsamp = test_preamble(est_to_dec, pktdata5)

    est_to_dec2, est_to_int2, pktdata6, re_cfo_2, re_cfo_freq_2 = fine_work(pktdata5)
    all_cfo_freq = re_cfo_freq + est_cfo_freq + re_cfo_freq_2
    detect, upsamp = test_preamble(est_to_dec2, pktdata6)
    pktdata7 = pktdata6[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]
    ans2n, ndatas = decode_payload(detect, est_to_dec, pktdata7, pkt_totcnt)

    # print('decoded data', ' '.join([f'{x:.0f}' for x in ans2n]))

    # ans1new = ans1n - np.array([est_cfo_slope * (i + Config.sfdpos + 2.5) for i in range(len(ans1r))])
    # print('fixed data  ', " ".join([f'{x:.3f}' for x in ans1new]))

    '''
    if Config.debug: print('est_to_dec / 8', est_to_dec / (Config.nsamp / Config.n_classes))
    ans1n += est_to_dec / (Config.nsamp / Config.n_classes)  # ???????????????'''  # TODO !!!
    angles = []
    avg1s = []
    avg2s = []
    ans2d = [x - int(x) for x in ans2n]
    ans1d = [x - int(x) for x in ans1n]


    ans2r = [round(x) % Config.n_classes for x in ans2n]
    if (ans2r != Config.symb_truth).any():
        return all_cfo_freq, angles, avg1s, avg2s, []


    angles0 = []
    for dataY in ndatas[detect + Config.sfdpos + 4:]:
        opts = Config
        dataX = cp.array(dataY)
        data1 = cp.matmul(Config.dataE1, dataX)
        data2 = cp.matmul(Config.dataE2, dataX)
        vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
        est = cp.argmax(vals)
        if est > 0:
            diff_avg0 = cmath.phase(data2[est] / data1[est])

            time_shift = int(est / opts.n_classes * opts.nsamp)
            time_split = opts.nsamp - time_shift
            data1 = dataY[:time_split] * opts.downchirp[time_shift:]
            data2 = dataY[time_split:] * opts.downchirp[:time_shift]
            avg1 = cmath.phase(np.sum(data1))
            avg2 = cmath.phase(np.sum(data2))
            avg1s.append(avg1)
            avg2s.append(avg2)
            diff_avg = cmath.phase(np.sum(data2) / np.sum(data1))

            angles.append(diff_avg)
            angles0.append(diff_avg0)
    #plt.plot(angles1, label='angles_w/_sfo')
    #plt.plot(angles0, '--', label='angles_w/o_sfo')
    #plt.legend()
    #plt.savefig(f'imgs/temp{pkt_totcnt}.jpg')
    #plt.clf()
    return all_cfo_freq, angles1, angles0, avg1s, avg2s, ans2d, ans1d


def decode_payload(detect, est_to_dec, pktdata4, pkt_totcnt):
    ndatas = pktdata4[: Config.symb_cnt * Config.nsamp].reshape(Config.symb_cnt, Config.nsamp)
    ans1, power1 = dechirp(ndatas[detect + Config.sfdpos + 2:], Config.downchirp)
    ans1n = ans1.get()
    ans1n += est_to_dec / 8

    if not min(power1) > np.mean(power1) / 2:
        print(f'power1 drops: {" ".join([str(round(x.item())) for x in power1])}')

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
    # print('preamble: ' + " ".join([f'{x:.3f}' for x in ans1]) + 'sfd: ' + " ".join([f'{x:.3f}' for x in ans2]), est_to_dec / 8)
    # print(power1)
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
    # print(' '.join([f'{x:.3f}' for x in ans1[: Config.preamble_len]]),
    #      ' '.join([f'{x:.3f}' for x in ans2[Config.sfdpos: Config.sfdpos + 2]]), sfd_upcode, sfd_downcode, re_cfo_0,
    #      est_to_0, detect)
    est_to_0 = est_to_0.get().item()
    re_cfo_0 = re_cfo_0.get().item()
    re_cfo_freq = re_cfo_0 * (Config.fs / fft_n)
    est_to_int = round(est_to_0)
    est_to_dec = est_to_0 - est_to_int
    pktdata3 = add_freq(pktdata2a, - re_cfo_freq)
    pktdata3 = np.roll(pktdata3, -est_to_int)
    return est_to_dec, est_to_int, pktdata3, re_cfo_0, re_cfo_freq


# read packets from file
if __name__ == "__main__":
    angless = []
    angles1s = []
    angles0s = []
    cfofreqs = []
    avg1ss = []
    avg2ss = []
    ans2ds = []
    ans1ds = []

    plt.rcParams['font.size'] = 15
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(8, 6))
    for file_path in os.listdir('.'):
        print(file_path)
        if not file_path.endswith('.bin'): continue
        Config.file_path = file_path
        pkt_cnt = 0
        pktdata = []
        fsize = int(os.stat(Config.file_path).st_size / (Config.nsamp * 4 * 2))
        if Config.debug: print(f'reading file: {Config.file_path} SF: {Config.sf} pkts in file: {fsize}')

        power_eval_len = 5000
        nmaxs = np.zeros((power_eval_len,))
        with open(Config.file_path, "rb") as f:
            for i in range(power_eval_len):  # while True:
                try:
                    rawdata = np.fromfile(f, dtype=cp.complex64, count=Config.nsamp)
                except EOFError:
                    if Config.debug: print("file complete")
                    break
                if len(rawdata) < Config.nsamp:
                    if Config.debug: print("file complete", len(rawdata))
                    break
                nmaxs[i] = np.max(np.abs(rawdata))
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(nmaxs.reshape(-1, 1))
        thresh = np.mean(kmeans.cluster_centers_)
        # counts, bins = np.histogram(nmaxs, bins=100)
        # print(counts, bins, kmeans.cluster_centers_, thresh)

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
                        all_cfo_freq, angles1, angles0, avg1s, avg2s, ans2d, ans1d = work(pkt_totcnt, cp.concatenate(pktdata))
                        angless.extend(angles0)
                        angles1s.append(angles1[0])
                        angles0s.append(angles0[0])
                        cfofreqs.append(all_cfo_freq)
                        avg1ss.extend(avg1s)
                        avg2ss.extend(avg2s)
                        ans2ds.extend(ans2d)
                        ans1ds.extend(ans1d)
                        color = [colorsys.hls_to_rgb((hue + 0.5) / 3, 0.4, 1) for hue in [0, 1, 2]]

                        name = 'angles'
                        # Draw Figs
                        data = angless
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
                            pickle.dump(angless, g)
                        plt.clf()

                        data = angles1s
                        count, bins_count = np.histogram(data, range=(-np.pi, np.pi), bins=100)
                        pdf = count / sum(count)
                        cdf = np.cumsum(pdf)
                        plt.plot(bins_count[1:], cdf, label=name)
                        plt.xlim(-np.pi, np.pi)
                        plt.xlabel('Angle (rad)')
                        plt.ylabel('Frequency')
                        plt.legend()
                        plt.savefig(name + '1s.pdf')
                        plt.savefig(name + '1s.png')
                        with open(name + '1s.pkl', 'wb') as g:
                            pickle.dump(angles1s, g)
                        plt.clf()

                        data = angles0s
                        count, bins_count = np.histogram(data, range=(-np.pi, np.pi), bins=100)
                        pdf = count / sum(count)
                        cdf = np.cumsum(pdf)
                        plt.plot(bins_count[1:], cdf, label=name)
                        plt.xlim(-np.pi, np.pi)
                        plt.xlabel('Angle (rad)')
                        plt.ylabel('Frequency')
                        plt.legend()
                        plt.savefig(name + '0s.pdf')
                        plt.savefig(name + '0s.png')
                        with open(name + '0s.pkl', 'wb') as g:
                            pickle.dump(angles0s, g)
                        plt.clf()

                        plt.scatter(range(len(cfofreqs)), cfofreqs, s=0.5)
                        plt.savefig(name + 'cfofreqs.png')
                        with open(name + 'cfofreqs.pkl', 'wb') as g:
                            pickle.dump(cfofreqs, g)
                        plt.clf()

                        plt.scatter(range(len(avg1ss[-2000:])), avg1ss[-2000:], s=0.1)
                        plt.savefig(name + 'avg1ss.png')
                        with open(name + 'avg1ss.pkl', 'wb') as g:
                            pickle.dump(avg1ss, g)
                        plt.clf()


                        plt.scatter(range(len(avg2ss[-2000:])), avg2ss[-2000:], s=0.1)
                        plt.savefig(name + 'avg2ss.png')
                        with open(name + 'avg2ss.pkl', 'wb') as g:
                            pickle.dump(avg2ss, g)
                        plt.clf()

                        plt.scatter(range(len(ans2ds)), ans2ds, s=0.1)
                        plt.savefig(name + 'ans2ds.png')
                        with open(name + 'ans2ds.pkl', 'wb') as g:
                            pickle.dump(ans2ds, g)
                        plt.clf()

                        plt.scatter(range(len(ans1ds)), ans1ds, s=0.1)
                        plt.savefig(name + 'ans1ds.png')
                        with open(name + 'ans1ds.pkl', 'wb') as g:
                            pickle.dump(ans1ds, g)
                        plt.clf()


                        pkt_totcnt += 1
                    pkt_cnt += 1
                    pktdata = []
                else:
                    pktdata.append(rawdata)
