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

def dechirp(ndata, refchirp, upsamp=None):
    if len(ndata.shape)==1: ndata = ndata.reshape(1, -1)
    global opts
    if not upsamp: upsamp = opts.fft_upsamp
    upsamp = opts.fft_upsamp #!!!
    chirp_data = ndata * refchirp
    ans = cp.zeros(ndata.shape[0], dtype=cp.float64)
    power = cp.zeros(ndata.shape[0], dtype=cp.float64)
    for idx in range(ndata.shape[0]):
        fft_raw = fft.fft(chirp_data[idx], n=opts.nsamp * upsamp, plan=opts.plan)
        target_nfft = opts.n_classes * upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(cp.float64) / upsamp
        power[idx] = cp.max(dat)
        # print(cp.argmax(dat), upsamp, ans[idx])
    return ans,power

cp.cuda.Device(1).use()
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
parser.add_argument('--pkt_len', type=int, default=129, help='Preamble Upchirp numbers.')
parser.add_argument('--file_path', type=str, default='4.dat', help='Preamble Upchirp numbers.')
parser.add_argument('--debug',action='store_false', help='Preamble Upchirp numbers.')
opts = parser.parse_args()
print('reading file', opts.file_path)


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
opts.plan = fft.get_fft_plan(cp.zeros(opts.nsamp * opts.fft_upsamp, dtype=cp.complex128))
# Example usage
with open(opts.file_path, 'rb') as f:
    pktdata = cp.fromfile(f, dtype=cp.complex64)
print(f'reading file: {opts.file_path} len: {len(pktdata)/opts.nsamp:.3f} symbols SF {opts.sf}')

pktdata = cp.concatenate((cp.zeros((opts.nsamp * 2 + 1000,)), pktdata))
pktdata = cp.concatenate(( pktdata, cp.zeros((opts.nsamp * 2 + 1000,))))

# pktdata = cp.array(rawdata[::2], dtype=cp.cfloat) + cp.array(rawdata[1::2], dtype=cp.cfloat) * 1j
pktdata /= cp.max(cp.abs(pktdata))

symb_cnt = len(pktdata)//opts.nsamp
ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)
if opts.debug: print('the shape of reshaped data in file', ndatas.shape)

for detect_loop in range(2):
    upsamp = 1 + detect_loop * (opts.fft_upsamp - 1)
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
    if opts.debug: print(f'detect packet at {detect}th window, preamble piece {ans1[detect: detect + opts.preamble_len]} \
                         {power1[detect: detect + opts.preamble_len]}, downpiece  {ans2[detect + opts.sfdpos : detect + opts.sfdpos + 2]}\
                         {power2[detect + opts.sfdpos : detect + opts.sfdpos + 2]}, ansval {ansval}, time shift {tshift}')

    sfd_upcode = ansval
    ansval2 = cp.angle(cp.sum(cp.exp(1j * 2 * cp.pi / opts.n_classes * ans2[detect + opts.sfdpos : detect + opts.sfdpos + 2]))) / (2 * cp.pi) * opts.n_classes 

    sfd_downcode = ansval2 

    est_cfo = (sfd_upcode + sfd_downcode) / 2
    est_to = -(sfd_upcode - sfd_downcode) / 2 # estimated time offset at the downcode
    re_cfo = est_cfo / opts.n_classes * opts.bw
    re_to = est_to / opts.n_classes * (opts.nsamp / opts.fs)
    print(f'{sfd_upcode} {sfd_downcode} ans: in samples cfo: {est_cfo} to: {est_to} reality: cfo {re_cfo} hz, to {re_to*1000} ms')
    re_sfo = re_cfo / opts.freq_sig * opts.fs
    re_sfo_t = (opts.nsamp / (opts.fs - re_sfo)) - (opts.nsamp / opts.fs)
    est_sfo_t = re_sfo_t / (opts.nsamp / opts.fs) * opts.n_classes
    print('re_sfo', re_sfo, 'est_sfo_t', est_sfo_t)

    cfosymb = cp.exp(- 2j * np.pi * re_cfo * cp.arange(0, len(pktdata) * 1 / opts.fs, 1 / opts.fs))

    pktdata *= cfosymb
    pktdata = cp.roll(pktdata, - round(est_to.item() * (opts.nsamp / opts.n_classes))) 
    print('rolling', - round(est_to.item() * (opts.nsamp / opts.n_classes)))

    ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)

    ans1, power1 = dechirp(ndatas[detect: detect + opts.sfdpos], opts.downchirp)
    ans2, power2 = dechirp(ndatas[detect + opts.sfdpos : detect + opts.sfdpos + 2], opts.upchirp)
    print(detect + opts.sfdpos)
    print('after cfo to correction: ans1\n',' '.join([f'{x:.3f}' for x in ans1]))
    print('after cfo to correction: power1\n',' '.join([f'{x:.3f}' for x in power1]))
    print('after cfo to correction: ans2\n',' '.join([f'{x:.3f}' for x in ans2]))
    print('after cfo to correction: power2\n',' '.join([f'{x:.3f}' for x in power2]))

pktdata = cp.roll(pktdata, - opts.nsamp * (detect + opts.sfdpos + 2) - opts.nsamp // 4)

ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)
ans1, power1 = dechirp(ndatas[detect + opts.sfdpos + 2:], opts.downchirp)
print('after cfo to correction: ans1\n',' '.join([f'{x:.3f}' for x in ans1]))
print('after cfo to correction: power1\n',' '.join([f'{x:.3f}' for x in power1]))
for i in range(len(power1)): 
    if power1[i] < opts.nsamp / 8: 
        ndatas = ndatas[:i]
        ans1 = ans1[:i]
        power1 = power1[:i]
        break
print('crop level', opts.nsamp / 8)
print('after cfo to correction: ans1\n',' '.join([f'{x:.3f}' for x in ans1]))
print('after cfo to correction: power1\n',' '.join([f'{x:.3f}' for x in power1]))
# print('after cfo to correction: power1\n',' '.join([f'{x:.3f}' for x in power1]))
print(len(ans1))
ans1n = ans1.get()
ans1r = [x - int(x) for x in ans1n]
print('res: ans1\n',' '.join([f'{x - int(x):.3f}' for x in ans1n]))
ans1u = np.unwrap(ans1r, period=1)
print('unwrap: ans1\n',' '.join([f'{x - int(x):.3f}' for x in ans1u]))
slope, intercept, r, p, std_err = stats.linregress(list(range(len(ans1r))), ans1u)
print('fit', slope, intercept,  r, p, std_err)
ans1new = [x - int(x) for x in ans1n] - np.array([slope * i + intercept for i in range(len(ans1r))])
print('after SFO correction: ans1new\n',' '.join([f'{x:.3f}' for x in ans1new]))
print('fit', slope,'re_sfo', re_sfo, 'est_sfo_t', est_sfo_t)

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
    est0.append(val2)


