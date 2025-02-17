import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
import sys

from utils import *

def objective_linear(cfofreq, time_error, pktdata2a):
    if time_error < 0: return -cfofreq, -time_error
    detect_symb_concat = gen_refchirp(cfofreq, time_error, len(pktdata2a))
    tint = math.ceil(time_error)
    # logger.warning(f"ObjLinear {cfofreq=} {time_error=} {tint=} {len(pktdata2a)=}")
    # fig = FigureResampler(go.Figure(layout_title_text=f"OL fit {cfofreq=:.3f} {time_error=:.3f}"))
    # fig.add_trace(go.Scatter(y=tocpu(cp.unwrap(cp.angle(pktdata2a[tint:tint + len(detect_symb_concat)])))))
    # fig.add_trace(go.Scatter(y=tocpu(cp.unwrap(cp.angle(detect_symb_concat)))))
    # fig.show()
    ress2 = -cp.unwrap(cp.angle(pktdata2a)) + cp.unwrap(cp.angle(detect_symb_concat))


    phasediff = ress2#cp.unwrap(cp.angle(cp.array(ress2)))
    est_dfreqs = []
    for fit_symbidx in range(0, Config.preamble_len):
        x_values = np.arange(Config.nsamp * fit_symbidx + tint + 50, Config.nsamp * (fit_symbidx + 1) + tint - 50)
        y_values = tocpu(phasediff[x_values])
        coefficients = np.polyfit(x_values, y_values, 1)
        est_dfreq = coefficients[0] * Config.fs / 2 / np.pi
        est_dfreqs.append(est_dfreq)
        # print(f"fitted curve {est_dfreq=:.2f} Hz")

    est_ups = []
    for fit_symbidx in range(Config.sfdpos, Config.sfdpos + 2):
        x_values = np.arange(Config.nsamp * fit_symbidx + tint + 50, Config.nsamp * (fit_symbidx + 1) + tint - 50)
        y_values = tocpu(phasediff[x_values])
        coefficients = np.polyfit(x_values, y_values, 1)
        est_ufreq = coefficients[0] * Config.fs / 2 / np.pi
        est_ups.append(est_ufreq)
        # print(f"fitted curve {est_ufreq=:.2f} Hz")
    ret_ufreq = np.mean(est_ups)
    ret_dfreq = np.mean(est_dfreqs[Config.skip_preambles:])
    if abs(cfofreq - -41774.0) < 3000:
        fig = px.scatter(y=est_dfreqs, title=f"objlinear {cfofreq:.3f} {time_error:.3f}")
        fig.add_hline(y=ret_dfreq)
        fig.show()

    beta = Config.bw / ((2 ** Config.sf) / Config.bw) / Config.fs
    ret_freq = (ret_ufreq + ret_dfreq)/2
    ret_tdiff = (ret_ufreq - ret_dfreq)/2 / beta
    logger.warning(f"linear {ret_freq=} {ret_tdiff=}")

    return ret_freq, ret_tdiff


def gen_matrix2(dt, est_cfo_f):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * Config.fs * (1 + 2 * est_cfo_f / Config.sig_freq)

    df = (est_cfo_f + dt * betai)
    cfosymb = cp.exp(2j * cp.pi * df * cp.linspace(0, Config.nsamp / Config.fs, num=Config.nsamp, endpoint=False)).astype(cp.complex64)
    decode_matrix_a = cp.zeros((Config.n_classes, Config.nsamp), dtype=cp.complex64)
    decode_matrix_b = cp.zeros((Config.n_classes, Config.nsamp), dtype=cp.complex64)
    tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1] * (1 - est_cfo_f / Config.sig_freq)
    for code in range(Config.n_classes):
        nsamples = around(Config.nsamp / Config.n_classes * (Config.n_classes - code))
        refchirp = mychirp(tstandard, f0=Config.bw * (-0.5 + code / Config.n_classes) * (1 + est_cfo_f / Config.sig_freq), f1=Config.bw * (0.5 + code / Config.n_classes) * (1 + est_cfo_f / Config.sig_freq),
                           t1=2 ** Config.sf / Config.bw * (1 - est_cfo_f / Config.sig_freq) )
        decode_matrix_a[code, :nsamples] = cp.conj(refchirp[:nsamples]) * cfosymb[:nsamples]

        refchirp = mychirp(tstandard, f0=Config.bw * (-1.5 + code / Config.n_classes) * (1 + est_cfo_f / Config.sig_freq), f1=Config.bw * (-0.5 + code / Config.n_classes) * (1 + est_cfo_f / Config.sig_freq),
                           t1=2 ** Config.sf / Config.bw * (1 - est_cfo_f / Config.sig_freq) )
        decode_matrix_b[code, nsamples:] = cp.conj(refchirp[nsamples:])* cfosymb[nsamples:]
    return decode_matrix_a, decode_matrix_b

def objective_decode(est_cfo_f, est_to_s, pktdata_in):
    codes = []
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * (1 + 2 * est_cfo_f / Config.sig_freq)
    for pidx in range(Config.sfdpos + 2, Config.total_len):
        start_pos_all_new = 2 ** Config.sf / Config.bw * Config.fs * (pidx + 0.25) * (1 - est_cfo_f / Config.sig_freq) + est_to_s
        start_pos = around(start_pos_all_new)
        tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
        dt = (start_pos - start_pos_all_new) / Config.fs
        dataX = pktdata_in[start_pos: Config.nsamp + start_pos] * cp.exp(-1j * 2 * cp.pi * (est_cfo_f + betai * dt) * tstandard)
        data1 = cp.matmul(Config.decode_matrix_a, dataX)
        data2 = cp.matmul(Config.decode_matrix_b, dataX)
        vals = cp.abs(data1) ** 2 + cp.abs(data2) ** 2
        coderet = cp.argmax(vals).item()
        codes.append(coderet)
    return codes
def find_power(est_cfo_f, est_to_s, pktdata_in):
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs * (1 - est_cfo_f / Config.sig_freq)
    powers = []
    px = []
    for pidx in range(- math.floor(est_to_s / nsamp_small), Config.sfdpos + 2):
        start_pos_all_new = nsamp_small * (pidx + 0) + est_to_s
        start_pos = around(start_pos_all_new)
        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
        powers.append(cp.mean(cp.abs(sig1)))
        px.append(pidx)

    for pidx in range(Config.sfdpos + 2, math.floor((len(pktdata_in) - est_to_s)/nsamp_small - 0.25) - 1):
        start_pos_all_new = nsamp_small * (pidx + 0.25) + est_to_s
        start_pos = around(start_pos_all_new)
        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
        powers.append(cp.mean(cp.abs(sig1)))
        px.append(pidx)
    data_reshaped = np.array(sqlist(powers)).reshape(-1, 1)

    # Apply KMeans with 2 clusters
    kmeans = KMeans(n_clusters=2,  n_init=10, random_state=42)
    kmeans.fit(data_reshaped)

    # Get the cluster labels
    labels = kmeans.labels_

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # Plot the data and the cluster centers
    # plt.scatter(powers, np.zeros_like(powers), c=labels, cmap='viridis')
    # plt.scatter(centers, np.zeros_like(centers), color='red', marker='x', s=100, label="Cluster Centers")
    # plt.title('K-means Clustering of the Data')
    # plt.xlabel('Data Points')
    # plt.legend()
    # plt.show()

    # Printing the clusters
    cluster_big = np.array(px)[labels == (centers[0] < centers[1])]
    cluster_small = np.array(px)[labels == (centers[0] > centers[1])]

    new_est_to_s = min(cluster_big) * nsamp_small + est_to_s
    totlen = len(cluster_big)
    if min(cluster_big) != 0 or totlen != Config.total_len:
        if min(cluster_big) != 0:
            logger.error(f"ERR find_power: {est_to_s=} misalign by {min(cluster_big)} {new_est_to_s=}")
        else:
            logger.error(f"ERR find_power: {Config.total_len=} != {totlen=}")
        str1 = ''.join([f'{x:6d}' for x in px])
        str2 = ''.join([f'{x:6.3f}' for x in powers])
        str3 = ''.join([f'{int(x in cluster_big):6d}' for x in px])
        logger.warning(f"ERR find_power:\n {str1} \n {str2} \n {str3}")
        showpower(est_cfo_f, est_to_s, pktdata_in, 'old')
        showpower(est_cfo_f, new_est_to_s, pktdata_in, 'new')
    assert totlen == Config.total_len, f"find_power {Config.total_len=} != {totlen=}"
    return new_est_to_s

def optimize_1dfreq_fast(sig2, tsymbr, freq1, margin):
    def obj1(freq, xdata, ydata):
        return -cp.abs(ydata.dot(cp.exp(xdata * -1j * 2 * cp.pi * freq.item()))).item()
    result = minimize(obj1, tos(freq1), args=(tsymbr, sig2), bounds=[(freq1 - margin, freq1 + margin)]) #!!!
    return result.x[0], - result.fun / cp.sum(cp.abs(sig2))

# <<< PURE LOG PRINTING, SHOW POWER OF EACH SYMBOL TO DETERMINE IF ALIGNED WITH SFD >>>
def showpower(est_cfo_f, est_to_s, pktdata_in, name):
    betai = Config.bw / ((2 ** Config.sf) / Config.bw) * (1 + 2 * est_cfo_f / Config.sig_freq)
    fs = []
    start_pos_all_new = 2 ** Config.sf / Config.bw * Config.fs * (Config.sfdpos - 3) * (1 - est_cfo_f / Config.sig_freq) + est_to_s
    start_pos = around(start_pos_all_new)
    tstandard = cp.arange(start_pos,start_pos + Config.nsamp * 7)
    fig = go.Figure(layout_title_text=f"{name} all")
    fig.add_trace(go.Scatter(x=tocpu(tstandard / Config.fs), y=tocpu(cp.unwrap(cp.angle(pktdata_in[tstandard])))))
    fig.show()
    for pidx in range(Config.sfdpos - 3, Config.sfdpos + 3):
        start_pos_all_new = 2 ** Config.sf / Config.bw * Config.fs * pidx * (1 - est_cfo_f / Config.sig_freq) + est_to_s
        start_pos = around(start_pos_all_new)
        tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1] + (start_pos - start_pos_all_new) / Config.fs
        if pidx < Config.sfdpos:
            refchirp = cp.exp(-1j * 2 * cp.pi * ((Config.bw * -0.5 * (1 + est_cfo_f / Config.sig_freq) + est_cfo_f) * tstandard + 0.5 * betai * tstandard * tstandard))
        else:
            refchirp = cp.exp(-1j * 2 * cp.pi * ((Config.bw * 0.5 * (1 + est_cfo_f / Config.sig_freq) + est_cfo_f) * tstandard - 0.5 * betai * tstandard * tstandard))
        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
        sig2 = sig1 * refchirp
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        freq1 = cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[cp.argmax(cp.abs(data0))]
        freq, pow = optimize_1dfreq_fast(sig2, tstandard, freq1, Config.fs / Config.fft_n * 5)
        fig = go.Figure(layout_title_text=f"{name} {pidx - Config.sfdpos=} {pow=} {freq=}")
        fig.add_trace(go.Scatter(x = tocpu(tstandard), y=tocpu(cp.unwrap(cp.angle(sig1)))))
        fig.add_trace(go.Scatter(x = tocpu(tstandard), y=tocpu(cp.unwrap(cp.angle(cp.conj(refchirp) * cp.exp(1j * 2 * cp.pi * tstandard * freq))))))
        fig.show()
        if pidx == Config.sfdpos - 1 or pidx == Config.sfdpos: fs.append(freq)
        logger.warning(f"showpower() {name} {pidx - Config.sfdpos=} {pow=} {freq1=} {freq=}")
    if name == "new":
        delta_cfo_f = (fs[0] + fs[1]) / 2
        delta_to_s = (-fs[0] + fs[1]) / 2 / betai * Config.fs
        logger.warning(f"showpower() {fs[0]=} {fs[1]=} {delta_cfo_f=} {delta_to_s=}")
        showpower(est_cfo_f + delta_cfo_f, est_to_s + delta_to_s, pktdata_in, 'chk')
        sys.exit(2)



def objective_decode_baseline(est_cfo_f, est_to_s, pktdata_in):
    nsamp_small = 2 ** Config.sf / Config.bw * Config.fs
    codes = []
    tstandard = cp.linspace(0, Config.nsamp / Config.fs, Config.nsamp + 1)[:-1]
    refchirp = mychirp(tstandard, f0=Config.bw * 0.5 - est_cfo_f, f1=Config.bw * -0.5 - est_cfo_f, t1=2 ** Config.sf / Config.bw)
    # # <<< PRINT POWER OF EACH SYMBOL >>> #
    # for pidx in range(math.floor((len(pktdata_in) - est_to_s)/nsamp_small - 0.25) - 10, math.floor((len(pktdata_in) - est_to_s)/nsamp_small - 0.25)):
    #     start_pos_all_new = nsamp_small * (pidx + 0.25) + est_to_s
    #     start_pos = around(start_pos_all_new)
    #     sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
    #     logger.warning(f"{Config.total_len=} {pidx=} {cp.mean(cp.abs(sig1))=}")

    for pidx in range(Config.sfdpos + 2, Config.total_len):
        start_pos_all_new = nsamp_small * (pidx + 0.25) + est_to_s
        start_pos = around(start_pos_all_new)

        sig1 = pktdata_in[start_pos: Config.nsamp + start_pos]
        sig2 = sig1 * refchirp
        data0 = myfft(sig2, n=Config.fft_n, plan=Config.plan)
        freq = cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs))[cp.argmax(cp.abs(data0))]
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=cp.fft.fftshift(cp.fft.fftfreq(Config.fft_n, d=1 / Config.fs)), y=cp.abs(data0)))
        # fig.show()
        if freq < 0: freq += Config.bw
        codex = freq / Config.bw * 2 ** Config.sf
        code = around(codex)
        code %= 2 ** Config.sf
        codes.append(code)
    return codes


def gen_refchirp(est_to_s, estf, length):
    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    x = (cp.arange(length) - est_to_s) * (1 + estf / Config.sig_freq)
    yi = cp.zeros_like(x, dtype=np.complex64)
    bwnew = Config.bw * (1 + estf / Config.sig_freq)
    betanew = beta * (1 + 2 * estf / Config.sig_freq)
    for i in range(Config.preamble_len):
        mask = (i * Config.nsampf < x) & (x <= (i + 1) * Config.nsampf)
        yi[mask] = cp.exp(2j * cp.pi * (
                betanew / 2 * (x[mask] - i * Config.nsampf) ** 2 / Config.fs ** 2 + (- bwnew / 2 + estf) * (
                x[mask] - i * Config.nsampf) / Config.fs))
    for i in range(Config.sfdpos, Config.sfdpos + 3):
        mask = (i * Config.nsampf < x) & (x <= (i + 1) * Config.nsampf) & (x <= (Config.sfdpos + 2.25) * Config.nsampf)
        yi[mask] = cp.exp(2j * cp.pi * (
                -betanew / 2 * (x[mask] - i * Config.nsampf) ** 2 / Config.fs ** 2 + (bwnew / 2 + estf) * (
                x[mask] - i * Config.nsampf) / Config.fs))
    return yi


def objective_core_new(est_cfo_f, est_to_s, pktdata_in):
    deltaf = 0
    estf = est_cfo_f + deltaf
    start_pos_all_new = est_to_s
    start_pos = around(start_pos_all_new)

    yi = gen_refchirp(est_to_s, estf, len(pktdata_in))

    retvals = 0
    for i in range(Config.preamble_len):
        xv = cp.arange(around(est_to_s + i * Config.nsamp), around(est_to_s + (i+1) * Config.nsamp))
        retvals += cp.abs(pktdata_in[xv].dot(cp.conj(yi[xv]))) / len(xv)
        # if i>=Config.preamble_len - 2:
        #     logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {i=} {cp.abs(pktdata_in[xv].dot(cp.conj(yi[xv]))) / len(xv)}")
    for i in range(Config.sfdpos, Config.sfdpos + 3):
        xv = cp.arange(around(est_to_s + i * Config.nsamp), around(est_to_s + (i+1) * Config.nsamp))
        if i == Config.sfdpos + 2:
            xv = cp.arange(around(est_to_s + i * Config.nsamp), around(est_to_s + (i + 0.25) * Config.nsamp))
        retvals += cp.abs(pktdata_in[xv].dot(cp.conj(yi[xv]))) / len(xv)
        # logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {i=} {cp.abs(pktdata_in[xv].dot(cp.conj(yi[xv]))) / len(xv)}")
    # logger.warning(f"objcorenew {est_cfo_f=:11.3f} {est_to_s=:11.3f} {retvals=}")

    return retvals



