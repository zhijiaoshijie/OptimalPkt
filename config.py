import numpy as np
from scipy.signal import chirp
import cupy as cp
import cupyx.scipy.fft as fft


# noinspection SpellCheckingInspection
class Config:
    sf = 7
    file_path = f'/data/djl/LoRaDatasetNew/LoRaDataNew/sf7-1.bin'
    bw = 125e3
    fs = 1e6
    n_classes = 2 ** sf
    nsamp = round(n_classes * fs / bw)

    freq_sig = 433e6
    preamble_len = 10
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
