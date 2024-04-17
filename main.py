import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import chirp
# from cupy.fft import fft
import cupyx.scipy.fft as fft
import cupy
import cupy as cp
import itertools
import os
import sys
from array import array
from scipy import stats
from tqdm import tqdm
import shutil



cp.cuda.Device(1).use()

def dechirp(ndata, refchirp, upsamp=None):
    if len(ndata.shape)==1: ndata = ndata.reshape(1, -1)
    global opts
    if not upsamp: upsamp = opts.fft_upsamp
    # upsamp = opts.fft_upsamp #!!!
    chirp_data = ndata * refchirp
    ans = cp.zeros(ndata.shape[0], dtype=cp.float64)
    power = cp.zeros(ndata.shape[0], dtype=cp.float64)
    for idx in range(ndata.shape[0]):
        fft_raw = fft.fft(chirp_data[idx], n=opts.nsamp * upsamp, plan=opts.plans[upsamp])
        target_nfft = opts.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(cp.float64) / upsamp
        power[idx] = cp.max(dat)
        # print(cp.argmax(dat), upsamp, ans[idx])
    return ans,power

parser = argparse.ArgumentParser(description="Example")
parser.add_argument('--sf', type=int, default=11, help='The spreading factor.')
# parser.add_argument('--bw', type=int, default=203125, help='The bandwidth.')
parser.add_argument('--bw', type=int, default=125000, help='The bandwidth.')
parser.add_argument('--fs', type=int, default=1000000, help='The sampling rate.')
# parser.add_argument('--freq_sig', type=int, default=2490e6, help='The sampling rate.')
parser.add_argument('--freq_sig', type=int, default=470e6, help='The sampling rate.')
# parser.add_argument('--preamble_len', type=int, default=6, help='Preamble Upchirp numbers.')
parser.add_argument('--preamble_len', type=int, default=10, help='Preamble Upchirp numbers.')
parser.add_argument('--code_len', type=int, default=2, help='Preamble Upchirp numbers.')
parser.add_argument('--fft_upsamp', type=int, default=4096, help='Preamble Upchirp numbers.')
# parser.add_argument('--pkt_len', type=int, default=48, help='Preamble Upchirp numbers.')
# parser.add_argument('--pkt_len', type=int, default=144, help='Preamble Upchirp numbers.')
parser.add_argument('--file_path', type=str, default='/data/djl/LoRaDatasetNew/LoRaDataNew/sf11_10.bin', help='Input file path.')
parser.add_argument('--debug',action='store_false', help='Preamble Upchirp numbers.')
opts = parser.parse_args()


opts.n_classes = 2 ** opts.sf
opts.nsamp = round(opts.fs * opts.n_classes / opts.bw)
opts.sfdpos = opts.preamble_len + opts.code_len

t = np.linspace(0, opts.nsamp / opts.fs, opts.nsamp+1)[:-1]
chirpI1 = chirp(t, f0=-opts.bw / 2, f1=opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=90)
chirpQ1 = chirp(t, f0=-opts.bw / 2, f1=opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=0)
opts.upchirp = cp.array(chirpI1 + 1j * chirpQ1)

chirpI1 = chirp(t, f0=opts.bw / 2, f1=-opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=90)
chirpQ1 = chirp(t, f0=opts.bw / 2, f1=-opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=0)
opts.downchirp = cp.array(chirpI1 + 1j * chirpQ1)
opts.plans = {1: fft.get_fft_plan(cp.zeros(opts.nsamp * 1, dtype=cp.complex128)),
opts.fft_upsamp: fft.get_fft_plan(cp.zeros(opts.nsamp * opts.fft_upsamp, dtype=cp.complex128))}




def work(pktdata):
    #pktdata = cp.concatenate((cp.zeros((opts.nsamp * 2 + 1000,)), pktdata))
    #pktdata = cp.concatenate(( pktdata, cp.zeros((opts.nsamp * 2 + 1000,))))

# pktdata = cp.array(rawdata[::2], dtype=cp.cfloat) + cp.array(rawdata[1::2], dtype=cp.cfloat) * 1j
    pktdata /= cp.max(cp.abs(pktdata))

    symb_cnt = opts.sfdpos + 5  #len(pktdata)//opts.nsamp
    ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)

    for detect_loop in range(2):
        upsamp = (1, opts.fft_upsamp)[detect_loop]
        ans1, power1 = dechirp(ndatas, opts.downchirp, upsamp)
        ans2, power2 = dechirp(ndatas, opts.upchirp, upsamp)
        vals = cp.zeros((symb_cnt, ), dtype=cp.float64)
        for i in range(symb_cnt - (opts.sfdpos + 2)):
            power = cp.sum(power1[i : i + opts.preamble_len]) + cp.sum(power2[i + opts.sfdpos : i + opts.sfdpos + 2])
            ans = cp.abs(cp.sum(cp.exp(1j * 2 * cp.pi / opts.n_classes * ans1[i : i + opts.preamble_len])))
            vals[i] = power * ans
        detect = cp.argmax(vals)

        ansval = cp.angle(cp.sum(cp.exp(1j * 2 * cp.pi / opts.n_classes * ans1[detect + 1: detect + opts.preamble_len - 1]))) / (2 * cp.pi) * opts.n_classes 
        # left or right may not be full symbol, detection may be off by a symbol
        tshift = round(ansval.item() * (opts.fs / opts.bw))
        if True: print(f'''detect packet at {detect}th window
        preamble {ans1[detect: detect + opts.preamble_len]} 
        power {power1[detect: detect + opts.preamble_len]}
        SFD {ans2[detect + opts.sfdpos : detect + opts.sfdpos + 2]}
        bins {power2[detect + opts.sfdpos : detect + opts.sfdpos + 2]}
        upcode {ansval}, time shift {tshift}''')

        sfd_upcode = ansval.get()
        ansval2 = cp.angle(cp.sum(cp.exp(1j * 2 * cp.pi / opts.n_classes * ans2[detect + opts.sfdpos : detect + opts.sfdpos + 2]))) / (2 * cp.pi) * opts.n_classes 

        sfd_downcode = ansval2.get()

        re_cfo = (sfd_upcode + sfd_downcode) / 2 / opts.n_classes * opts.bw # estimated CFO, in Hz
        # cfo positive: cause preamble +, sfd +
        est_to = (sfd_downcode - sfd_upcode) / 2 * (opts.nsamp / opts.n_classes)  # estimated time offset at the downcode, in samples
        # to positive (signal moved right, 0s at left): cause preamble -, sfd +
        # cfo - to = upcode
        # cfo + to = downcode
        # cfo = (upcode + downcode) / 2, in bins
        # to = (downcode - upcode) / 2, in bins
        if est_to < 0: est_to += opts.nsamp
        est_sfo_t = - opts.nsamp * (1 - opts.freq_sig / (opts.freq_sig + re_cfo))
        # len(received) - len(standard), in samples, plus = longer
        # cfo positive -> sender frequency higher -> signal shorter -> sfo negative

        cfosymb = cp.exp(- 2j * np.pi * re_cfo * cp.linspace(0, (len(pktdata) - 1) / opts.fs, num=len(pktdata)))


        pktdata *= cfosymb
        est_to_int = round(est_to)
        est_to_dec = est_to - est_to_int
        pktdata = pktdata[est_to_int:]
        symb_cnt = len(pktdata)//opts.nsamp

        ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)

        ans1, power1 = dechirp(ndatas[detect: detect + opts.sfdpos], opts.downchirp)
        ans2, power2 = dechirp(ndatas[detect + opts.sfdpos : detect + opts.sfdpos + 2], opts.upchirp)

    # packet contents
    pktdata = pktdata[opts.nsamp * (detect + opts.sfdpos + 2) + opts.nsamp // 4 :]
    symb_cnt = len(pktdata)//opts.nsamp

    ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)
    ans1, power1 = dechirp(ndatas[detect + opts.sfdpos + 2:], opts.downchirp)
    # print('power1:\n',' '.join([f'{x:.3f}' for x in power1]))
    ans1n = ans1.get()
    ans1r = [x - int(x) for x in ans1n]
    ans1u = np.unwrap(ans1r, period=1)
    slope, intercept, r, p, std_err = stats.linregress(list(range(len(ans1r))), ans1u)
    ans1new = [x - int(x) for x in ans1n] - np.array([slope * i + intercept for i in range(len(ans1r))])
    if opts.debug: print(f'''detection: {detect}th cfo: {(sfd_upcode + sfd_downcode) / 2} bins {re_cfo} Hz
 to: {-(sfd_upcode - sfd_downcode) / 2} bins {est_to} samples
 sfo: {est_sfo_t} samples fit: {slope} samples''')
    print(' '.join([':.3f' % x for x in ans1u]))
    print(' '.join([':.3f' % x for x in ans1new]))
    print("-" * shutil.get_terminal_size()[0])
    '''
    sys.exit(1)

    ans1[ans1 > opts.n_classes // 2] -= opts.n_classes
    ans2[ans2 > opts.n_classes // 2] -= opts.n_classes
    slope, intercept, r, p, std_err = stats.linregress(list(range(opts.preamble_len)), ans1[:opts.preamble_len].get())

    est_sfo = slope
    sfd_upcode = slope * opts.sfdpos + intercept
    if opts.debug: print(f'sfo detection: slope {slope} intercept {intercept}')

    re_sfo = re_cfo / opts.freq_sig * opts.fs
    re_sfo_t = (opts.nsamp / (opts.fs - re_sfo)) - (opts.nsamp / opts.fs)
    est_sfo_t = re_sfo_t / (opts.nsamp / opts.fs) * opts.n_classes
    print('est_sfo_t', est_sfo_t)



    est0 = []
    print((len(pktdata)- tshift)//opts.nsamp)

#for symbid in range((len(pktdata)-tshift)//opts.nsamp-1):
    for symbid in range(opts.pkt_len):
        sfdshift = 0.25
        if symbid < opts.preamble_len+opts.code_len+2: sfdshift = 0

        pkt_tshift = (est_to + est_sfo_t * (symbid + sfdshift - (opts.preamble_len+opts.code_len))) * (opts.fs / opts.bw)
        pkt_tshift_int = cupy.round_(pkt_tshift)
        pkt_tshift_dec = pkt_tshift - pkt_tshift_int
        tstart = opts.nsamp * symbid + int(sfdshift * opts.nsamp) + pkt_tshift_int

        ndata = pktdata[tstart : tstart + opts.nsamp]

        if symbid in range(opts.preamble_len+opts.code_len, opts.preamble_len+opts.code_len + 2):
            val, power = dechirp(ndata, opts.upchirp)
            power *= -1
        else:
            val, power = dechirp(ndata, opts.downchirp)
        tmod = pkt_tshift_dec / (opts.fs / opts.bw)
        if symbid in range(opts.preamble_len+opts.code_len, opts.preamble_len+opts.code_len + 2): tmod *= -1
        val2 = val - est_cfo + tmod
        print("{:3d} \t {:10.5f} \t {:10.5f} \t  {:10.5f} \t {:10.5f}".format(symbid, val, val - est_cfo, val2, power))
        print("{:8.5f}".format(tmod))
        print("{:8d}".format(pkt_tshift_dec))
        est0.append(val2)'''

# read packets from file
thresh = 0.0005
pkt_cnt = 0
pktdata = []
assert os.path.isfile(opts.file_path), "file_path not found"
fsize = int(os.stat(opts.file_path).st_size / (opts.nsamp * 4 * 2))
print(f'reading file: {opts.file_path} SF: {opts.sf} pkts in file: {fsize}')
nmaxs = []
if False:
    with open(opts.file_path, "rb") as f:
        while True:
            try:
                rawdata = np.fromfile(f, dtype=cp.complex64, count=opts.nsamp)
            except EOFError:
                print("file complete")
                break
            if len(rawdata) < opts.nsamp:
                print("file complete", len(rawdata))
                break
            nmaxs.append(np.max(np.abs(rawdata)))
    counts, bins = np.histogram(nmaxs, bins=100)
    print(counts, bins)
with open(opts.file_path, "rb") as f:
    while True:
        try:
            rawdata = cp.fromfile(f, dtype=cp.complex64, count=opts.nsamp)
        except EOFError:
            print("file complete")
            break
        if len(rawdata) < opts.nsamp:
            print("file complete", len(rawdata))
            break
        nmax = cp.max(cp.abs(rawdata))

        if nmax < thresh:
            if len(pktdata) > 14 and pkt_cnt > 20:
                print(f"start parsing pkt len: {len(pktdata)}")
                work(cp.concatenate(pktdata))
            pktdata = []
            pkt_cnt += 1
        else: pktdata.append(rawdata)
