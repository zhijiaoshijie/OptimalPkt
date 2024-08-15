import colorsys
import os
import pickle
import shutil
import cmath
import math
import cupy as cp
import cupyx.scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
from tqdm import tqdm

cp.cuda.Device(0).use()


class Config:
    sf = 7
    bw = 125e3
    fs = 1e6
    n_classes = 2 ** sf
    nsamp = round(n_classes * fs / bw)

    freq_sig = 470e6
    preamble_len = 8
    code_len = 2
    fft_upsamp = 4096
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


opts = Config()
Config = Config()
dataE1 = cp.zeros((opts.n_classes, opts.nsamp), dtype=cp.cfloat)
dataE2 = cp.zeros((opts.n_classes, opts.nsamp), dtype=cp.cfloat)
for symbol_index in range(opts.n_classes):
    time_shift = int(symbol_index / opts.n_classes * opts.nsamp)
    time_split = opts.nsamp - time_shift
    dataE1[symbol_index][:time_split] = opts.downchirp[time_shift:]
    if symbol_index != 0: dataE2[symbol_index][time_split:] = opts.downchirp[:time_shift]


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


# noinspection SpellCheckingInspection
def work(pktdata_in):
    pktdata = pktdata_in / cp.max(cp.abs(pktdata_in))

    symb_cnt = Config.sfdpos + 5  # len(pktdata)//Config.nsamp
    ndatas = pktdata[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)

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

    ansval = cp.angle( cp.sum(cp.exp(1j * 2 * cp.pi / Config.n_classes * ans1[detect + 1: detect + Config.preamble_len - 1]))) / ( 2 * cp.pi) * Config.n_classes

    sfd_upcode = ansval.get()
    ansval2 = cp.angle(cp.sum( cp.exp(1j * 2 * cp.pi / Config.n_classes * ans2[detect + Config.sfdpos: detect + Config.sfdpos + 2]))) / ( 2 * cp.pi) * Config.n_classes

    sfd_downcode = ansval2.get()
    re_cfo_0 = (sfd_upcode + sfd_downcode) / 2
    re_cfo = re_cfo_0 / Config.n_classes * Config.bw  # estimated CFO, in Hz

    est_to = (sfd_downcode - sfd_upcode) / 2 * (
                Config.nsamp / Config.n_classes)  # estimated time offset at the downcode, in samples
    if abs(re_cfo + 24867) > Config.bw / 8:  # !!! TODO test the preamble to see if it is zero
        re_cfo -= Config.bw / 2
        est_to -= Config.nsamp / 2
    if abs(re_cfo) > Config.bw / 4:
        print('!' * shutil.get_terminal_size()[0])
    if est_to < 0: est_to += Config.nsamp

    cfosymb = cp.exp(- 2j * np.pi * re_cfo * cp.linspace(0, (len(pktdata) - 1) / Config.fs, num=len(pktdata)))

    pktdata2a = pktdata * cfosymb
    est_to_int = round(est_to)
    est_to_dec = est_to - est_to_int
    pktdata2a = pktdata2a[est_to_int:]  # !!! TODO est_to_dec

    symb_cnt = len(pktdata2a) // Config.nsamp
    if symb_cnt < 30: # !!! TODO make it a parameter
        print('Too short signal')
        return []

    pktdata2 = pktdata2a[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]
    symb_cnt = len(pktdata2) // Config.nsamp # TODO length of symbol is varying

    ndatas = pktdata2[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)
    ans1, power1 = dechirp(ndatas[detect + Config.sfdpos + 2:], Config.downchirp)
    ans1n = ans1.get()

    if Config.debug: print('est_to_dec / 8', est_to_dec / (Config.nsamp / Config.n_classes))
    ans1n += est_to_dec / (Config.nsamp / Config.n_classes)  # ???????????????

    if Config.debug: print('pktdata2', ' '.join([str(round(x.item())) for x in ans1n]))

    angles = []
    for dataY in ndatas[detect + Config.sfdpos + 4:]:
        opts = Config
        dataX = cp.array(dataY)
        data1 = cp.abs(cp.matmul(dataE1, dataX))
        data2 = cp.abs(cp.matmul(dataE2, dataX))
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
