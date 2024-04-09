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
    global opts
    if not upsamp: upsamp = opts.fft_upsamp
    chirp_data = ndata * refchirp
    fft_raw = fft(chirp_data, len(chirp_data) * upsamp)
    target_nfft = opts.n_classes * upsamp

    cut1 = cp.array(fft_raw[:target_nfft])
    cut2 = cp.array(fft_raw[-target_nfft:])
    dat = abs(cut1)+abs(cut2)
    return cp.argmax(dat).item() / upsamp,   cp.max(dat)/cp.max(cp.abs(ndata))
cp.cuda.Device(0).use()
parser = argparse.ArgumentParser()
parser.add_argument('--sf', type=int, default=12, help='The spreading factor.')
parser.add_argument('--bw', type=int, default=203125, help='The bandwidth.')
# parser.add_argument('--bw', type=int, default=125000, help='The bandwidth.')
parser.add_argument('--fs', type=int, default=1000000, help='The sampling rate.')
parser.add_argument('--preamble_len', type=int, default=8, help='Preamble Upchirp numbers.')
parser.add_argument('--code_len', type=int, default=0, help='Preamble Upchirp numbers.')
parser.add_argument('--fft_upsamp', type=int, default=1024, help='Preamble Upchirp numbers.')
parser.add_argument('--pkt_len', type=int, default=48, help='Preamble Upchirp numbers.')
opts = parser.parse_args()


opts.n_classes = 2 ** opts.sf
opts.nsamp = int(opts.fs * opts.n_classes / opts.bw)

t = np.linspace(0, opts.nsamp / opts.fs, opts.nsamp+1)[:-1]
chirpI1 = chirp(t, f0=-opts.bw / 2, f1=opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=90)
chirpQ1 = chirp(t, f0=-opts.bw / 2, f1=opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=0)
opts.upchirp = cp.array(chirpI1 + 1j * chirpQ1)

chirpI1 = chirp(t, f0=opts.bw / 2, f1=-opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=90)
chirpQ1 = chirp(t, f0=opts.bw / 2, f1=-opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=0)
opts.downchirp = cp.array(chirpI1 + 1j * chirpQ1)


# Example usage
file_path = '9.dat'
with open(file_path, 'rb') as f:
    rawdata = np.fromfile(f, dtype=np.float32)
print(len(rawdata)/opts.nsamp)
print(opts.n_classes)

pktdata = cp.array(rawdata[::2], dtype=cp.cfloat) + cp.array(rawdata[1::2], dtype=cp.cfloat) * 1j
est0 = []
for symbid in range(opts.preamble_len*2):
    ndata = pktdata[opts.nsamp * symbid: opts.nsamp * (symbid + 1)]
    ans, power = dechirp(ndata, opts.downchirp, 1)
    if ans > opts.n_classes/2: ans -= opts.n_classes
    if power > opts.nsamp/2: est0.append(ans)
print('z',est0)

symbid = 5
ndata = pktdata[opts.nsamp * symbid : opts.nsamp * (symbid + 1)]
tshift = round(dechirp(ndata, opts.downchirp)[0] * (opts.fs / opts.bw))
print(dechirp(ndata, opts.downchirp))
print(tshift)
pktdata = cp.roll(pktdata, tshift)

est0 = []
for symbid in range(opts.preamble_len*2):
    ndata = pktdata[opts.nsamp * symbid: opts.nsamp * (symbid + 1)]
    ans, power = dechirp(ndata, opts.downchirp)
    if ans > opts.n_classes/2: ans -= opts.n_classes
    if power > opts.nsamp/2: est0.append(ans)
print('l',est0)

'''
pktdata = cp.roll(pktdata, - tshift*2)
est0 = []
for symbid in range(opts.preamble_len*2):
    ndata = pktdata[opts.nsamp * symbid: opts.nsamp * (symbid + 1)]
    ans, power = dechirp(ndata, opts.downchirp, 1)
    if power > opts.nsamp/2: est0.append(ans)
print('r',est0)

est0 = []
for symbid in range(opts.preamble_len):
    ndata = pktdata[opts.nsamp * symbid : opts.nsamp * (symbid + 1)]
    est0.append(dechirp(ndata, opts.upchirp))
print(est0)
est0 = []
for symbid in range(opts.preamble_len):
    ndata = pktdata[opts.nsamp * symbid : opts.nsamp * (symbid + 1)]
    est0.append(dechirp(ndata, opts.downchirp))
print(est0)

est0 = [x[0] for x in est0]'''
est0 = est0[:opts.preamble_len]
slope, intercept, r, p, std_err = stats.linregress(list(range(len(est0))), est0)
est_sfo = slope
sfd_upcode = slope * (opts.preamble_len+opts.code_len) + intercept
print(slope, intercept, 'sl')
if sfd_upcode > opts.n_classes // 2: sfd_upcode -= opts.n_classes
print(sfd_upcode)

sfd_codes = []    
# for symbid in range(opts.preamble_len+opts.code_len, opts.preamble_len+opts.code_len+2):
for symbid in range(opts.preamble_len * 2):
    ndata = pktdata[opts.nsamp * symbid : opts.nsamp * (symbid + 1)]
    ans, power = dechirp(ndata, opts.upchirp)
    if power > opts.nsamp/2: sfd_codes.append(ans)
print('sfd codes', sfd_codes)
sfd_downcode = (sfd_codes[0] + sfd_codes[1] + est_sfo) / 2
if sfd_downcode > opts.n_classes // 2: sfd_downcode -= opts.n_classes

print(sfd_upcode, sfd_downcode)

est_cfo = (sfd_upcode + sfd_downcode) / 2
est_to = -(sfd_upcode - sfd_downcode) / 2
re_cfo = est_cfo / opts.n_classes * opts.bw
re_to = est_to / opts.n_classes * (opts.nsamp / opts.fs)
print('ans',tshift, est_sfo, est_cfo, est_to, 'cfo',re_cfo,'hz,to', re_to*1000,'ms')


cfosymb = cp.array(np.exp(2j * np.pi * -re_cfo * t))

# pktdata *= cfosymb
pktdata = cp.roll(pktdata,-round(est_to * (opts.nsamp // opts.n_classes))) 

est0 = []
for symbid in range(opts.preamble_len*2):
    ndata = pktdata[opts.nsamp * symbid: opts.nsamp * (symbid + 1)]
    ans, power = dechirp(ndata, opts.downchirp)
    if ans > opts.n_classes/2: ans -= opts.n_classes
    if power > opts.nsamp/2: est0.append(ans)
print('l',est0)




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
