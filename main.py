import os
import shutil

import matplotlib.pyplot as plt
from scipy import stats

import numpy as np
from scipy.signal import chirp
import cupy as cp
import cupyx.scipy.fft as fft
from tqdm import tqdm
import sys
cp.cuda.Device(0).use()


class Config:
    sf = 7
    file_path = f'sf7-470-new-test.bin'
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


# noinspection SpellCheckingInspection
def work(pktdata):
    pktdata /= cp.max(cp.abs(pktdata))

    symb_cnt = Config.sfdpos + 5  # len(pktdata)//Config.nsamp
    ndatas = pktdata[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)

    ndatatest = Config.downchirp
    shiftt = 8 * 3
    ndatatest = np.concatenate((ndatatest[shiftt:], ndatatest[:shiftt]))
    ans1, power1 = dechirp(ndatatest, Config.upchirp)
    # print(ans1)

    for detect_loop in range(1):
        # upsamp = (1, Config.fft_upsamp)[detect_loop]
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

        ansval = cp.angle(
            cp.sum(cp.exp(1j * 2 * cp.pi / Config.n_classes * ans1[detect + 1: detect + Config.preamble_len - 1]))) / (
                         2 * cp.pi) * Config.n_classes
        # left or right may not be full symbol, detection may be off by a symbol
        tshift = round(ansval.item() * (Config.fs / Config.bw))

        sfd_upcode = ansval.get()
        ansval2 = cp.angle(
            cp.sum(cp.exp(
                1j * 2 * cp.pi / Config.n_classes * ans2[detect + Config.sfdpos: detect + Config.sfdpos + 2]))) / (
                          2 * cp.pi) * Config.n_classes

        sfd_downcode = ansval2.get()
        if False:
            print(f'''detect packet at {detect}th window
            preamble {ans1[detect: detect + Config.preamble_len]} 
            power {power1[detect: detect + Config.preamble_len]}
            SFD {ans2[detect + Config.sfdpos: detect + Config.sfdpos + 2]}
            bins {power2[detect + Config.sfdpos: detect + Config.sfdpos + 2]}
            upcode {sfd_upcode}, downcode {sfd_downcode} time shift {tshift}''')

            print('preamble', [round(x.get().item()) for x in ans1[detect: detect + Config.preamble_len]],
            'sfd', [round(x.get().item()) for x in ans2[detect + Config.sfdpos: detect + Config.sfdpos + 2]])
        print(sfd_upcode, sfd_downcode, ' '.join([str(round(x.get().item())) for x in ans1[detect: detect + Config.preamble_len]]), ' '.join([str(round(x.get().item())) for x in ans2[detect + Config.sfdpos: detect + Config.sfdpos + 2]]))

        re_cfo = (sfd_upcode + sfd_downcode) / 2 / Config.n_classes * Config.bw  # estimated CFO, in Hz

        # !!!
        if abs(re_cfo + 24867) > Config.bw / 8: 
            re_cfo -= Config.bw / 2
        if abs(re_cfo) > Config.bw / 4: 
            print('!' * shutil.get_terminal_size()[0])
        # cfo positive: cause preamble +, sfd +
        est_to = (sfd_downcode - sfd_upcode) / 2 * (
                Config.nsamp / Config.n_classes)  # estimated time offset at the downcode, in samples
        # to positive (signal moved right, 0s at left): cause preamble -, sfd +
        # cfo - to = upcode
        # cfo + to = downcode
        # cfo = (upcode + downcode) / 2, in bins
        # to = (downcode - upcode) / 2, in bins
        if est_to < 0: est_to += Config.nsamp

        # est_sfo_t = - Config.nsamp * (1 - Config.freq_sig / (Config.freq_sig + re_cfo))
        est_sfo_t = - Config.n_classes * (1 - Config.freq_sig / (Config.freq_sig + re_cfo))  # ???????????????????

        # len(received) - len(standard), in samples, plus = longer
        # cfo positive -> sender frequency higher -> signal shorter -> sfo negative

        cfosymb = cp.exp(- 2j * np.pi * re_cfo * cp.linspace(0, (len(pktdata) - 1) / Config.fs, num=len(pktdata)))

        pktdata *= cfosymb
        est_to_int = round(est_to)
        est_to_dec = est_to - est_to_int
        pktdata = pktdata[est_to_int:] #!!!
        symb_cnt = len(pktdata) // Config.nsamp

        ndatas = pktdata[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)

        ans1, power1 = dechirp(ndatas[detect: detect + Config.sfdpos], Config.downchirp)
        ans2, power2 = dechirp(ndatas[detect + Config.sfdpos: detect + Config.sfdpos + 2], Config.upchirp)
        if Config.debug: print(f'''===detection: {detect}th cfo: {(sfd_upcode + sfd_downcode) / 2} bins {re_cfo} Hz
         to: {-(sfd_upcode - sfd_downcode) / 2} bins {est_to} samp sfo: {est_sfo_t} samp\n''')  # samp sfo seems to be larger by 8x

    # est_to_dec # !!! !!! !!!  !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!! !!!
    # packet contents
    # pktdata = cp.concatenate((pktdata[Config.nsamp * detect : Config.nsamp * (detect + Config.preamble_len)], pktdata[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4 :]))
    preamble1 = pktdata[Config.nsamp * (detect + 4): Config.nsamp * (detect + 5)]
    pktdata = pktdata[Config.nsamp * (detect + Config.sfdpos + 2) + Config.nsamp // 4:]
    symb_cnt = len(pktdata) // Config.nsamp

    ndatas = pktdata[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)
    fixeddata = ndatas[detect + Config.sfdpos + 2 + 5]
    ans1, power1 = dechirp(ndatas[detect + Config.sfdpos + 2:], Config.downchirp)
    # if Config.debug: print('power1:\n',' '.join([f'{x:.3f}' for x in power1]))
    ans1n = ans1.get()

    if Config.debug: print('est_to_dec / 8', est_to_dec / (Config.nsamp / Config.n_classes))
    # print('est_to_dec / 8', est_to_dec / (Config.nsamp / Config.n_classes))
    ans1n += est_to_dec / (Config.nsamp / Config.n_classes)  # ???????????????

    ans1r = [x - int(x) for x in ans1n]
    ans1u = np.unwrap(ans1r, period=1)
    slope, intercept, r, p, std_err = stats.linregress(list(range(len(ans1r))), ans1u)
    ans1new = ans1n - np.array([(slope * i + intercept) - int(slope * i + intercept) for i in range(len(ans1r))])
    if Config.debug: print(f'fit: {slope} samples')

    #     est_sfo_t = - Config.n_classes * (1 - Config.freq_sig / (Config.freq_sig + re_cfo)) # ???????????????????
    freq_sig_est = re_cfo / (1 / (1 - slope / Config.n_classes) - 1)
    # print('estimated frequency:', freq_sig_est/1e6, 'mhz')

    '''
    print('pktdata', ' '.join([str(round(x.item())) for x in ans1n]))
    print(f'{len(ans1n)=}')
    print("-" * shutil.get_terminal_size()[0])
    print(freq_sig_est, re_cfo, slope)'''



    '''
    if Config.debug: print('pktdata power', ' '.join([f'{x:.3f}' % x for x in power1]))
    if Config.debug: print("-" * shutil.get_terminal_size()[0])
    if Config.debug: print('pktdata: unwrapped', ' '.join([f'{x:.3f}' % x for x in ans1u]))
    if Config.debug: print("-" * shutil.get_terminal_size()[0])
    if Config.debug: print('unwrapped - linregress', ' '.join([f'{x:.3f}' % x for x in ans1new]))
    if Config.debug: print("-" * shutil.get_terminal_size()[0])
    ans1new2 = ans1n - np.array(
        [(est_sfo_t * i + intercept) - int(est_sfo_t * i + intercept) for i in range(len(ans1r))])
    if Config.debug: print('unwrapped - est_sfo', ' '.join([f'{x:.3f}' % x for x in ans1new2]))
    if Config.debug: print("-" * shutil.get_terminal_size()[0])
    if Config.debug: print("-" * shutil.get_terminal_size()[0])
    '''

    return 0# freq_sig_est


# read packets from file
if __name__ == "__main__":
    thresh = 0.002
    pkt_cnt = 0
    pktdata = []
    assert os.path.isfile(Config.file_path), "file_path not found"
    fsize = int(os.stat(Config.file_path).st_size / (Config.nsamp * 4 * 2))
    if Config.debug: print(f'reading file: {Config.file_path} SF: {Config.sf} pkts in file: {fsize}')
    nmaxs = []
    with open(Config.file_path, "rb") as f:
        for i in tqdm(range(5000)):  # while True:
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
    estfreqs = []
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
                    res = work(cp.concatenate(pktdata))
                    if len(estfreqs) < 10 or abs(res - np.mean(estfreqs)) < np.std(estfreqs) * 3:
                        estfreqs.append(res)
                    else:
                        print(
                            f'rejected {pkt_totcnt}th val {res}, dist to {np.mean(estfreqs)} ({abs(res - np.mean(estfreqs))})> {np.std(estfreqs)}')
                    plt.scatter(range(len(estfreqs)), estfreqs)
                    plt.axhline(470e6, linestyle='--', color='black')
                    plt.axhline(433e6, linestyle='--', color='black')
                    plt.savefig('estfreqs.jpg')
                    plt.clf()
                    pkt_totcnt += 1
                pkt_cnt += 1
                pktdata = []
            else:
                pktdata.append(rawdata)
