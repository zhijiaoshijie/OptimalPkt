import logging
import os
import time

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

logger = logging.getLogger('my_logger')
level = logging.WARNING
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(level)  # Set the console handler level
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(level)  # Set the file handler level
formatter = logging.Formatter('%(asctime)s - %(message)s')
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

use_gpu = True
if use_gpu:
    import cupy as cp
    import cupyx.scipy.fft as fft
else:
    import numpy as cp
    import scipy.fft as fft


def togpu(x):
    if use_gpu and not isinstance(x, cp.ndarray):
        return cp.array(x)
    else:
        return x


def tocpu(x):
    if use_gpu and isinstance(x, cp.ndarray):
        return x.get()
    else:
        return x


def mychirp(t, f0, f1, t1):
    beta = (f1 - f0) / t1
    phase = 2 * cp.pi * (f0 * t + 0.5 * beta * t * t)
    sig = cp.exp(1j * togpu(phase)).astype(cp.complex64)
    return sig

# noinspection PyTypeChecker
def cp_str(x, precision=2, suppress_small=False):
    return np.array2string(tocpu(x), precision=precision, formatter={'float_kind': lambda k: f"{k:.3f}"},
                           floatmode='fixed', suppress_small=suppress_small, max_line_width=np.inf)


script_path = __file__
mod_time = os.path.getmtime(script_path)
readable_time = time.ctime(mod_time)
logger.warning(f"Last modified time of the script: {readable_time}")


class Config:
    # Set parameters
    sf = 11
    bw = 125e3
    fs = 1e6
    sig_freq = 470e6
    file_paths = ['/data/djl/datasets/sf11_240906_0.bin']
    fft_upsamp = 1024
    dataout_path = f'/data/djl/datasets/sf11_240906_out_{fft_upsamp}_test'
    payload_len_expected = 23  # num of payload symbols
    preamble_len = 8
    code_len = 2
    progress_bar_disp = True
    skip_pkts = 0

    # preprocess
    if not os.path.exists(dataout_path):
        os.makedirs(dataout_path)
        logger.warning(f'W00_OUTDIR: make output directory {dataout_path}')
    else:
        logger.warning(f"E00_OUTDIR: {dataout_path} already exists")
    # base_dir = '/data/djl/datasets/Dataset_50Nodes'
    # file_paths = []
    # for file_name in os.listdir(base_dir):
    #     if file_name.startswith('sf7') and file_name.endswith('.bin'):
    #         file_paths.append(os.path.join(base_dir, file_name))
    progress_bar = None
    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    nsamp = round(n_classes * fs / bw)
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3
    t = cp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    upchirp = mychirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw)
    downchirp = mychirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw)
    if use_gpu:
        plan = fft.get_fft_plan(cp.zeros(nsamp * fft_upsamp, dtype=cp.complex64))
    else:
        plan = None
    pkt_idx = 0


if use_gpu:
    cp.cuda.Device(0).use()
opts = Config()
Config = Config()


def myfft(chirp_data, n, plan):
    if use_gpu:
        return fft.fft(chirp_data, n=n, plan=plan)
    else:
        return fft.fft(chirp_data, n=n)


# noinspection SpellCheckingInspection
def dechirp(ndata, refchirp):
    if len(ndata.shape) == 1:
        ndata = ndata.reshape(1, -1)
    chirp_data = ndata * refchirp
    ans = cp.zeros(ndata.shape[0], dtype=float)
    power = cp.zeros(ndata.shape[0], dtype=float)
    for idx in range(ndata.shape[0]):
        fft_raw = myfft(chirp_data[idx], n=Config.nsamp * Config.fft_upsamp, plan=Config.plan)
        target_nfft = Config.n_classes * Config.fft_upsamp

        cut1 = cp.array(fft_raw[:target_nfft])
        cut2 = cp.array(fft_raw[-target_nfft:])
        dat = cp.abs(cut1) + cp.abs(cut2)
        ans[idx] = cp.argmax(dat).astype(float) / Config.fft_upsamp
        power[idx] = cp.max(dat)  # logger.debug(cp.argmax(dat), Config.fft_upsamp, ans[idx])
    return ans, power


def add_freq(pktdata_in, est_cfo_freq):
    cfosymb = cp.exp(2j * cp.pi * est_cfo_freq * cp.linspace(0, (len(pktdata_in) - 1) / Config.fs, num=len(pktdata_in)))
    cfosymb = cfosymb.astype(cp.complex64)
    pktdata2a = pktdata_in * cfosymb
    return pktdata2a


def coarse_work_fast(pktdata_in):
    fft_n = Config.nsamp * Config.fft_upsamp
    detect_range_pkts = 3
    phaseFlag = False
    fft_ups = cp.zeros((Config.preamble_len + detect_range_pkts, fft_n), dtype=cp.complex64)
    fft_downs = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.complex64)
    for pidx in range(Config.preamble_len + detect_range_pkts):
        sig1 = pktdata_in[Config.nsamp * pidx: Config.nsamp * (pidx + 1)] * Config.downchirp
        fft_ups[pidx] = myfft(sig1, n=fft_n, plan=Config.plan)
    for pidx in range(2 + detect_range_pkts):
        sig1 = (pktdata_in[Config.nsamp * (pidx + Config.sfdpos): Config.nsamp * (pidx + Config.sfdpos + 1)]
                * Config.upchirp)
        fft_downs[pidx] = myfft(sig1, n=fft_n, plan=Config.plan)
    if not phaseFlag:
        fft_ups = cp.abs(fft_ups) ** 2
        fft_downs = cp.abs(fft_downs) ** 2

    fft_vals = cp.zeros((detect_range_pkts, 3), dtype=cp.float32)
    for pidx in range(detect_range_pkts):
        fft_val_up = cp.argmax(cp.abs(cp.sum(fft_ups[pidx: pidx + Config.preamble_len], axis=0)))
        if fft_val_up > fft_n / 2: fft_val_up -= fft_n
        fft_val_down = cp.argmax(cp.abs(cp.sum(fft_downs[pidx: pidx + 2], axis=0)))  # + fft_down_lst[pidx]))
        if fft_val_down > fft_n / 2: fft_val_down -= fft_n
        fft_val_abs = cp.max(cp.abs(cp.sum(fft_ups[pidx: pidx + Config.preamble_len], axis=0))) \
                      + cp.max(cp.abs(cp.sum(fft_downs[pidx: pidx + 2], axis=0)))  # + fft_down_lst[pidx]))
        # logger.info(f"{fft_val_up=} {fft_val_down=} {fft_n=}")
        est_cfo_r = (fft_val_up + fft_val_down) / 2 / fft_n  # rate, [0, 1)
        est_to_r = (fft_val_down - fft_val_up) / 2 / fft_n  # rate, [0, 1)
        if abs(est_cfo_r - 1 / 2) <= 1 / 4:  # abs(cfo) > 1/4
            est_cfo_r += 1 / 2
            est_to_r += 1 / 2
        est_cfo_r %= 1  # [0, 1)
        est_to_r %= 1  # [0, 1)
        if est_cfo_r > 1 / 2: est_cfo_r -= 1  # [-1/2, 1/2)
        if est_to_r > 1 / 2:
            est_to_r -= 1  # [-1/2, 1/2)
            if pidx == 0: fft_val_abs *= 0  # shift left is nothing!
        est_cfo_f = est_cfo_r * Config.fs
        est_to_s = (est_to_r * 8 + pidx) * Config.nsamp  # add detect packet pos TODO
        if abs(est_to_r) > 1/8:
            logger.error(f"E07_LARGE_TIME_OFFSET: {Config.pkt_idx=} {fft_val_up=} {fft_val_down=} {est_cfo_r=} {est_to_r=} {est_cfo_f=} {est_to_s=} {pidx=}")
        if abs(est_cfo_f) >= Config.fs / 8:
            logger.warning(f"E07_LARGE_CFO: {Config.pkt_idx=} {fft_val_up=} {fft_val_down=} {est_cfo_r=} {est_to_r=} {est_cfo_f=} {est_to_s=} {pidx=}")
        fft_vals[pidx] = cp.array((est_cfo_f, est_to_s, fft_val_abs), dtype=cp.float32)
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(y=fft_ups[idx].get(), mode='lines', name='Ours'))
        # fig.add_trace(go.Scatter(y=fft_downs[idx].get(), mode='lines', name='Ours'))
        # fig.update_layout(title=f"{idx=} {fft_val_up=} {fft_val_down=}")
        # fig.show()
    # sys.exit(0)
    bestidx = cp.argmax(fft_vals[:, 2])
    # if bestidx != 0:
    #     logger.warning(f"E05_BESTIDX_NOT_ZERO: {bestidx=}")
    #     logger.warning(f"E05_BESTIDX_NOT_ZERO: fft_ups={cp_str(cp.argmax(cp.abs(fft_ups), axis=1)/fft_n)}")
    #     logger.warning(f"E05_BESTIDX_NOT_ZERO: fft_dns={cp_str(cp.argmax(cp.abs(fft_downs), axis=1)/fft_n)}")
    #     logger.warning(f"E05_BESTIDX_NOT_ZERO: fft_lls={cp_str(cp.argmax(cp.abs(fft_down_lst), axis=1)/fft_n)}")
    # logger.info(cp.argmax(fft_vals[:, 2]))
    return fft_vals[bestidx][0].item(), fft_vals[bestidx][1].item()


def test_work_coarse(pktdata_in):
    est_cfo_f2, est_to_s2 = coarse_work_fast(pktdata_in)
    logger.info(f"I02_WORK_RESULT {est_cfo_f2=}, {est_to_s2=}")

    pktdata2 = add_freq(pktdata_in, - est_cfo_f2)
    est_to_int = round(est_to_s2)
    est_to_dec = est_to_s2 - est_to_int
    pktdata2 = cp.roll(pktdata2, - est_to_int)

    # pktdata4A = pktdata2[:Config.nsamp * Config.sfdpos]
    # ans1A, power1A = decode_payload(est_to_dec, pktdata4A)
    # logger.info(f'I03_1: Before SFO decode Preamble: {len(ans1A)=}\n    {cp_str(ans1A)=}\n    {cp_str(power1A)=}')
    # pktdata4B = pktdata2[int(Config.nsamp * (Config.sfdpos + 2 + 0.25)):]
    # ans1B, power1B = decode_payload(est_to_dec, pktdata4B)
    # logger.info(f'I03_2: Before SFO decode Payload : {len(ans1B)=}\n    {cp_str(ans1B)=}\n    {cp_str(power1B)=}')

    est_cfo_slope = est_cfo_f2 / Config.sig_freq * Config.bw * Config.fs / Config.nsamp
    sig_time = len(pktdata2) / Config.fs
    logger.info(f'I03_3: SFO {est_cfo_f2=} Hz, {est_cfo_slope=} Hz/s, {sig_time=} s')
    t = cp.linspace(0, sig_time, len(pktdata2) + 1)[:-1]
    est_cfo_symbol = mychirp(t, f0=0, f1=- est_cfo_slope * sig_time, t1=sig_time)
    pktdata2C = pktdata2 * est_cfo_symbol

    pktdata4A = pktdata2C[:Config.nsamp * Config.sfdpos]
    ans1A, power1A = decode_payload(est_to_dec, pktdata4A)
    logger.info(f'I03_4: After SFO decode Preamble: {len(ans1A)=}\n    {cp_str(ans1A)=}\n    {cp_str(power1A)=}')
    pktdata4B = pktdata2C[int(Config.nsamp * (Config.sfdpos + 2 + 0.25)):]
    ans1B, power1B = decode_payload(est_to_dec, pktdata4B)
    logger.info(f'I03_5: After SFO decode Payload : {len(ans1B)=}\n    {cp_str(ans1B)=}\n    {cp_str(power1B)=}')

    return ans1B, pktdata2C


def decode_payload(est_to_dec, pktdata4):
    symb_cnt = len(pktdata4) // Config.nsamp
    ndatas = pktdata4[: symb_cnt * Config.nsamp].reshape(symb_cnt, Config.nsamp)
    ans1n, power1 = dechirp(ndatas, Config.downchirp)
    ans1n += est_to_dec / 8
    if not min(power1) > cp.mean(power1) / 2:
        drop_idx = next((idx for idx, num in enumerate(power1) if num < cp.mean(power1) / 2), -1)
        logger.info(
            f'E01_POWER_DROP: {Config.pkt_idx=} power1 drops: {drop_idx=} {len(ans1n)=}\n    {cp_str(ans1n)=}\n    {cp_str(power1)=}')
        ans1n = ans1n[:drop_idx]
        power1 = power1[:drop_idx]
    return ans1n, power1


def read_large_file(file_path_in):
    with open(file_path_in, 'rb') as file:
        while True:
            try:
                rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp)
                Config.progress_bar.update(len(rawdata) * 8)
            except EOFError:
                logger.warning("E04_FILE_EOF: file complete with EOF")
                break
            if len(rawdata) < Config.nsamp:
                logger.warning(f"E05_FILE_FIN: file complete, {len(rawdata)=}")
                break
            yield rawdata


def read_pkt(file_path_in, threshold, min_length=20):
    current_sequence = []
    for rawdata in read_large_file(file_path_in):
        number = cp.max(cp.abs(rawdata))
        if number > threshold:
            current_sequence.append(rawdata)
        else:
            if len(current_sequence) > min_length:
                yield cp.concatenate(current_sequence)
            current_sequence = []


# read packets from file
if __name__ == "__main__":
    angles = []

    for file_path in Config.file_paths:
        pkt_cnt = 0
        pktdata = []
        fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
        logger.warning(f'W01_READ_START: reading file: {file_path} SF: {Config.sf} pkts in file: {fsize} {Config.skip_pkts=}')
        Config.progress_bar = tqdm(total=int(os.stat(file_path).st_size), unit='B', unit_scale=True, desc=file_path,
                                   disable=not Config.progress_bar_disp)

        power_eval_len = 5000
        nmaxs = cp.zeros(power_eval_len, dtype=float)
        for idx, rawdata in enumerate(read_large_file(file_path)):
            nmaxs[idx] = cp.max(cp.abs(rawdata))
            if idx == power_eval_len - 1: break
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(tocpu(nmaxs.reshape(-1, 1)))
        thresh = cp.mean(kmeans.cluster_centers_)
        counts, bins = cp.histogram(nmaxs, bins=100)
        logger.debug(f"D00_CLUSTER: cluster: {kmeans.cluster_centers_[0]} {kmeans.cluster_centers_[1]} {thresh=}")
        threshpos = np.searchsorted(tocpu(bins), thresh).item()
        logger.debug(f"D00_CLUSTER: lower: {cp_str(counts[:threshpos])}")
        logger.debug(f"D00_CLUSTER: higher: {cp_str(counts[threshpos:])}")

        for Config.pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=20)):
            if Config.pkt_idx <= Config.skip_pkts: continue
            Config.progress_bar.set_description(os.path.splitext(os.path.basename(file_path))[0] + ':' + str(Config.pkt_idx))
            logger.info(f"W02_READ_PKT_START: {Config.pkt_idx=} {len(pkt_data)=} {len(pkt_data)/Config.nsamp=}")
            pkt_data_0 = cp.concatenate((cp.zeros(Config.nsamp // 2, dtype=cp.complex64), pkt_data,
                                         cp.zeros(Config.nsamp // 2, dtype=cp.complex64)))
            _, pkt_data_A = test_work_coarse(pkt_data_0 / cp.mean(cp.abs(pkt_data_0)))
            pkt_data_B = cp.concatenate((cp.zeros(Config.nsamp // 2, dtype=cp.complex64), pkt_data_A,
                                         cp.zeros(Config.nsamp // 2, dtype=cp.complex64)))
            ans_list, pkt_data_C = test_work_coarse(pkt_data_B)
            payload_data = pkt_data_C[int(Config.nsamp * (Config.sfdpos + 2 + 0.25)):]
            if len(ans_list) != Config.payload_len_expected:
                logger.warning(
                    f"E03_ANS_LEN: {Config.pkt_idx=} {len(pkt_data)=} {len(pkt_data)/Config.nsamp=} {len(ans_list)=}")
            else:
                outpath = os.path.join(Config.dataout_path, 'part' + str(Config.pkt_idx // 1000), str(Config.pkt_idx))
                if not os.path.exists(outpath): os.makedirs(outpath)
                for idx, decode_ans in enumerate(list(tocpu(ans_list))):
                    data = payload_data[Config.nsamp * idx: Config.nsamp * (idx + 1)]
                    data.tofile(os.path.join(outpath,
                                             f"{idx}_{round(decode_ans) % Config.n_classes}_{Config.pkt_idx}_{Config.sf}.mat"))
