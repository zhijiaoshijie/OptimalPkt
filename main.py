import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import chirp
from cupy.fft import fft
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
    chirp_data = ndatas * refchirp
    fft_raw = fft(chirp_data, n=opts.nsamp * upsamp, axis = 1)
    target_nfft = opts.n_classes * upsamp

    cut1 = cp.array(fft_raw[:, :target_nfft])
    cut2 = cp.array(fft_raw[:, -target_nfft:])
    dat = abs(cut1)+abs(cut2)
    ans = cp.argmax(dat, axis=1).astype(np.float64) / upsamp
    power = cp.max(dat, axis=1)
    return ans,power

cp.cuda.Device(0)
parser = argparse.ArgumentParser(description="Example")
parser.add_argument('--sf', type=int, default=12, help='The spreading factor.')
parser.add_argument('--bw', type=int, default=203125, help='The bandwidth.')
# parser.add_argument('--bw', type=int, default=125000, help='The bandwidth.')
parser.add_argument('--fs', type=int, default=1000000, help='The sampling rate.')
parser.add_argument('--preamble_len', type=int, default=6, help='Preamble Upchirp numbers.')
parser.add_argument('--code_len', type=int, default=2, help='Preamble Upchirp numbers.')
parser.add_argument('--fft_upsamp', type=int, default=8192, help='Preamble Upchirp numbers.')
parser.add_argument('--pkt_len', type=int, default=48, help='Preamble Upchirp numbers.')
parser.add_argument('--file_path', type=str, default='9.dat', help='Preamble Upchirp numbers.')
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


# Example usage
with open(opts.file_path, 'rb') as f:
    rawdata = np.fromfile(f, dtype=np.float32)
print(f'reading file: {opts.file_path} len: {len(rawdata)/opts.nsamp:.3f} symbols SF {opts.sf}')

pktdata = cp.array(rawdata[::2], dtype=cp.cfloat) + cp.array(rawdata[1::2], dtype=cp.cfloat) * 1j
pktdata /= np.max(np.abs(pktdata))

symb_cnt = len(pktdata)//opts.nsamp
ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)
if opts.debug: print('the shape of reshaped data in file', ndatas.shape)

ans1, power1 = dechirp(ndatas, opts.downchirp, 1)
ans2, power2 = dechirp(ndatas, opts.upchirp, 1)
vals = cp.zeros((symb_cnt, ), dtype=np.float64)
for i in range(symb_cnt - (opts.sfdpos + 2)):
    power = cp.sum(power1[i : i + opts.preamble_len]) + cp.sum(power2[i + opts.sfdpos : i + opts.sfdpos + 2])
    ans = cp.abs(cp.sum(cp.exp(1j * 2 * cp.pi / opts.n_classes * ans1[i : i + opts.preamble_len])))
    vals[i] = power * ans
detect = cp.argmax(vals)

ansval = cp.angle(cp.sum(cp.exp(1j * 2 * cp.pi / opts.n_classes * ans1[detect + 1: detect + opts.preamble_len - 1]))) / (2 * cp.pi) * opts.n_classes 
# left or right may not be full symbol, detection may be off by a symbol
tshift = round(ansval.item() * (opts.fs / opts.bw))
if opts.debug: print(f'detect packet at {detect}th window, preamble piece {ans1[detect: detect + opts.preamble_len]} {power1[detect: detect + opts.preamble_len]}, downpiece  {ans2[detect + opts.sfdpos : detect + opts.sfdpos + 2]} {power2[detect + opts.sfdpos : detect + opts.sfdpos + 2]}, ansval {ansval}, time shift {tshift}')
pktdata = cp.roll(pktdata, tshift - opts.nsamp * detect)


ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)

# fine-grained
ans1, power1 = dechirp(ndatas, opts.downchirp)
ans2, power2 = dechirp(ndatas, opts.upchirp)
if opts.debug: print(ans1)
ans1[ans1 > opts.n_classes // 2] -= opts.n_classes
ans2[ans2 > opts.n_classes // 2] -= opts.n_classes
slope, intercept, r, p, std_err = stats.linregress(list(range(opts.preamble_len)), ans1[:opts.preamble_len].get())

est_sfo = slope
sfd_upcode = slope * opts.sfdpos + intercept
if opts.debug: print(f'sfo detection: slope {slope} intercept {intercept} sfd_upcode {sfd_upcode}')

sfd_downcode = (ans2[opts.sfdpos] + ans2[opts.sfdpos + 1] + est_sfo) / 2

est_cfo = (sfd_upcode + sfd_downcode) / 2
est_to = -(sfd_upcode - sfd_downcode) / 2
re_cfo = est_cfo / opts.n_classes * opts.bw
re_to = est_to / opts.n_classes * (opts.nsamp / opts.fs)
print(f'ans: in samples sfo: {est_sfo} cfo: {est_cfo} to: {est_to} reality: cfo {re_cfo} hz, to {re_to*1000} ms')

cfosymb = cp.exp(- 2j * np.pi * re_cfo * cp.arange(0, len(pktdata) * 1 / opts.fs, 1 / opts.fs))

pktdata *= cfosymb
pktdata = cp.roll(pktdata, - round(est_to.item() * (opts.nsamp / opts.n_classes))) 

ndatas = pktdata[ : symb_cnt * opts.nsamp].reshape(symb_cnt, opts.nsamp)
print(ndatas.shape)
ans1, power1 = dechirp(ndatas, opts.downchirp)
ans2, power2 = dechirp(ndatas, opts.upchirp)
print(ans1, power1, ans2, power2)




'''
est0 = []
for symbid in range(opts.preamble_len+opts.code_len):
    ndata = pktdata[opts.nsamp * symbid : opts.nsamp * (symbid + 1)]
    est0.append(dechirp(ndata, opts.downchirp) - est_cfo - est_to - est_sfo * (symbid - (opts.preamble_len+opts.code_len)))
for symbid in range(opts.preamble_len+opts.code_len, opts.preamble_len+opts.code_len+2):
    ndata = pktdata[opts.nsamp * symbid : opts.nsamp * (symbid + 1)]
    est0.append(dechirp(ndata, opts.upchirp) - est_cfo + est_to + est_sfo * (symbid - (opts.preamble_len+opts.code_len)))
print(est0)'''

est0 = []
print((len(pktdata)- tshift)//opts.nsamp)

#for symbid in range((len(pktdata)-tshift)//opts.nsamp-1):
for symbid in range(opts.pkt_len):
    sfdshift = 0.25
    if symbid < opts.preamble_len+opts.code_len+2: sfdshift = 0

    pkt_tshift = (est_to + est_sfo * (symbid + sfdshift - (opts.preamble_len+opts.code_len))) * (opts.fs / opts.bw)
    pkt_tshift_int = round(pkt_tshift)
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
    print("{:3d} \t {:10.5f} \t {:10.5f} \t  {:10.5f} \t {:8.5f} \t {:8d} \t {:15.5f}".format(symbid, val, val - est_cfo, val2, tmod, tshift - round(pkt_tshift), power))
    est0.append(val2)
