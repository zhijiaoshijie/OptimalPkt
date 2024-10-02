import logging
import os
import shutil
import time

import math
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# for debug
# import platform

test_mode = False
local_mode = False

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
if local_mode: use_gpu = False
if use_gpu:
    import cupy as cp
    import cupyx.scipy.fft as fft
else:
    import pyfftw
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


def mychirp(t, f0, f1, t1, t0=0, phase0=0):
    beta = (f1 - f0) / (t1 - t0)
    phase = 2 * cp.pi * (f0 * (t - t0) + 0.5 * beta * (t - t0) ** 2) + phase0
    sig = cp.exp(1j * togpu(phase)).astype(cp.complex64)
    return sig


def mysymb(t, f0, f1, t1, t0, symb, phase0=0, phase1=0):
    beta = (f1 - f0) / (t1 - t0)
    f0R = f0 + (f1 - f0) * (symb / Config.n_classes)
    tjump = t0 + (t1 - t0) * (1 - symb / Config.n_classes)
    phaseA = 2 * cp.pi * (f0R * (t - t0) + 0.5 * beta * (t - t0) ** 2) + phase0
    phaseA[t > tjump] = 0
    phaseB = 2 * cp.pi * ((f0R - (f1 - f0)) * (t - t0) + 0.5 * beta * (t - t0) ** 2) + phase1
    phaseB[t <= tjump] = 0
    phase = phaseA + phaseB
    sig = cp.exp(1j * togpu(phase)).astype(cp.complex64)
    return sig

# noinspection PyTypeChecker
def cp_str(x, precision=2, suppress_small=False):
    return np.array2string(tocpu(x), precision=precision, formatter={'float_kind': lambda k: f"{k:.3f}"},
                           floatmode='fixed', suppress_small=suppress_small, max_line_width=np.inf)


logger.warning(f"Last modified time of the script: {time.ctime(os.path.getmtime(__file__))}")


class Config:
    # Set parameters
    sf = 10
    bw = 125e3
    fs = 1e6
    sig_freq = 470e6
    file_paths = []

    if local_mode:
        bpath = 'D:\\desktop\\cut_woldro_3'
        cpath = os.path.join(bpath, f'sf{sf}')
        for x in os.listdir(cpath):
            file_paths.append(os.path.join(cpath, x))
        fft_upsamp = 1#024
        dataout_path = os.path.join(bpath, f'test_sf{sf}_wol_{fft_upsamp}_dataout')
    else:
        cpath = '/data/djl/datasets/240928_woldro'
        for x in os.listdir(cpath): file_paths.append(os.path.join(cpath, x))
        fft_upsamp = 1024
        dataout_path = os.path.join('/data/djl/datasets/', f'sf{sf}_wol_{fft_upsamp}_dataout')




    logger.warning(f"W00_STARTUP_INFO: {sf=} {fft_upsamp=} {file_paths[0]=} {len(file_paths)=} {dataout_path=}")
    payload_len_expected = 23  # num of payload symbols
    preamble_len = 8
    code_len = 2
    progress_bar_disp = not test_mode
    skip_pkts = 0

    # preprocess
    if not os.path.exists(dataout_path):
        os.makedirs(dataout_path)
        logger.warning(f'W00_OUTDIR: make output directory {dataout_path}')
    else:
        logger.warning(f"E00_OUTDIR: {dataout_path} already exists")
        shutil.rmtree(dataout_path)
    # base_dir = '/data/djl/datasets/Dataset_50Nodes'
    # file_paths = []
    # for file_name in os.listdir(base_dir):
    #     if file_name.startswith('sf7') and file_name.endswith('.bin'):
    #         file_paths.append(os.path.join(base_dir, file_name))
    progress_bar = None
    n_classes = 2 ** sf
    tsig = 2 ** sf / bw * fs  # in samples
    nsamp = round(n_classes * fs / bw)
    part_max_size = 1e9
    packets_per_part = int(part_max_size / nsamp / 8 / payload_len_expected)
    sfdpos = preamble_len + code_len
    sfdend = sfdpos + 3
    t = cp.linspace(0, nsamp / fs, nsamp + 1)[:-1]
    upchirp = mychirp(t, f0=-bw / 2, f1=bw / 2, t1=2 ** sf / bw)
    downchirp = mychirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw)
    if use_gpu:
        plan = fft.get_fft_plan(cp.zeros(nsamp * fft_upsamp, dtype=cp.complex64))
    else:
        plan = None

        fft_length = nsamp * fft_upsamp
        input_array = pyfftw.empty_aligned(fft_length, dtype='complex64')
        output_array = pyfftw.empty_aligned(fft_length, dtype='complex64')

        # Create the FFTW plan
        fft_plan = pyfftw.FFTW(input_array, output_array, direction='FFTW_FORWARD', flags=['FFTW_MEASURE'])

    pkt_idx = 0

    prtidx = 0

    fft_n = nsamp * fft_upsamp
    detect_range_pkts = 3
    fft_ups = cp.zeros((preamble_len + detect_range_pkts, fft_n), dtype=cp.complex64)
    fft_downs = cp.zeros((2 + detect_range_pkts, fft_n), dtype=cp.complex64)

if use_gpu:
    cp.cuda.Device(0).use()
opts = Config()
Config = Config()


def myfft(chirp_data, n, plan):
    if use_gpu:
        return fft.fft(chirp_data, n=n, plan=plan)
    else:
        np.copyto(Config.input_array[:len(chirp_data)], chirp_data)

        # Execute the FFT plan (in-place execution on input_array)
        Config.fft_plan()

        # Append the result to the results list
        return Config.output_array


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
    phaseFlag = False

    for pidx in range(Config.preamble_len + Config.detect_range_pkts):
        sig1 = pktdata_in[Config.nsamp * pidx: Config.nsamp * (pidx + 1)] * Config.downchirp
        Config.fft_ups[pidx] = myfft(sig1, n=Config.fft_n, plan=Config.plan)
    for pidx in range(2 + Config.detect_range_pkts):
        sig1 = (pktdata_in[Config.nsamp * (pidx + Config.sfdpos): Config.nsamp * (pidx + Config.sfdpos + 1)]
                * Config.upchirp)
        Config.fft_downs[pidx] = myfft(sig1, n=Config.fft_n, plan=Config.plan)
    if not phaseFlag:
        fft_ups2 = cp.abs(Config.fft_ups) ** 2
        fft_downs2 = cp.abs(Config.fft_downs) ** 2
    else:
        fft_ups2 = Config.fft_ups
        fft_downs2 = Config.fft_downs


    fft_vals = cp.zeros((Config.detect_range_pkts, 3), dtype=cp.float32)
    for pidx in range(Config.detect_range_pkts):
        fft_val_up = cp.argmax(cp.abs(cp.sum(fft_ups2[pidx: pidx + Config.preamble_len], axis=0)))
        if fft_val_up > Config.fft_n / 2: fft_val_up -= Config.fft_n
        fft_val_down = cp.argmax(cp.abs(cp.sum(fft_downs2[pidx: pidx + 2], axis=0)))  # + fft_down_lst[pidx]))
        if fft_val_down > Config.fft_n / 2: fft_val_down -= Config.fft_n
        fft_val_abs = cp.max(cp.abs(cp.sum(fft_ups2[pidx: pidx + Config.preamble_len], axis=0))) \
                      + cp.max(cp.abs(cp.sum(fft_downs2[pidx: pidx + 2], axis=0)))  # + fft_down_lst[pidx]))
        # logger.info(f"{fft_val_up=} {fft_val_down=} {Config.fft_n=}")
        est_cfo_r = (fft_val_up + fft_val_down) / 2 / Config.fft_n  # rate, [0, 1)
        est_to_r = (fft_val_down - fft_val_up) / 2 / Config.fft_n  # rate, [0, 1)
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
        # fig.add_trace(go.Scatter(y=fft_ups2[idx].get(), mode='lines', name='Ours'))
        # fig.add_trace(go.Scatter(y=fft_downs2[idx].get(), mode='lines', name='Ours'))
        # fig.update_layout(title=f"{idx=} {fft_val_up=} {fft_val_down=}")
        # fig.show()
    # sys.exit(0)
    bestidx = cp.argmax(fft_vals[:, 2])
    # print(fft_vals[:, 2])
    # if bestidx != 0:
    #     logger.warning(f"E05_BESTIDX_NOT_ZERO: {bestidx=}")
    #     logger.warning(f"E05_BESTIDX_NOT_ZERO: fft_ups2={cp_str(cp.argmax(cp.abs(fft_ups2), axis=1)/Config.fft_n)}")
    #     logger.warning(f"E05_BESTIDX_NOT_ZERO: fft_dns={cp_str(cp.argmax(cp.abs(fft_downs2), axis=1)/Config.fft_n)}")
    #     logger.warning(f"E05_BESTIDX_NOT_ZERO: fft_lls={cp_str(cp.argmax(cp.abs(fft_down_lst), axis=1)/Config.fft_n)}")
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
                Config.progress_bar.update(file.tell() - Config.progress_bar.n)
            except EOFError:
                logger.warning("E04_FILE_EOF: file complete with EOF")
                break
            if len(rawdata) < Config.nsamp:
                logger.warning(f"E05_FILE_FIN: file complete, {len(rawdata)=}")
                break
            yield rawdata


def read_pkt(file_path_in, threshold, min_length):
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
def main():

    oldtime = time.time()
    Config.file_pkt_idx = 0
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


        Config.progress_bar.reset()

        for Config.pkt_idx, pkt_data in enumerate(read_pkt(file_path, thresh, min_length=20)):
            if Config.pkt_idx <= Config.skip_pkts: continue

            # output indexes
            out_pkt_idx = Config.file_pkt_idx + Config.pkt_idx
            prtidx = 0
            while True:
                outprtpath = os.path.join(Config.dataout_path, 'part' + str(prtidx))
                if os.path.exists(outprtpath):
                    if len(os.listdir(outprtpath)) > Config.packets_per_part:
                        prtidx += 1
                    else:
                        break
                else:
                    break

            Config.progress_bar.set_description(os.path.splitext(os.path.basename(file_path))[0] + ':' + str(prtidx) + ':' + str(out_pkt_idx))
            logger.info(f"W02_READ_PKT_START: {Config.pkt_idx=} {len(pkt_data)=} {len(pkt_data)/Config.nsamp=}")
            pkt_data_0 = cp.concatenate((cp.zeros(Config.nsamp // 2, dtype=cp.complex64), pkt_data,
                                         cp.zeros(Config.nsamp // 2, dtype=cp.complex64)))
            _, pkt_data_A = test_work_coarse(pkt_data_0 / cp.mean(cp.abs(pkt_data_0)))
            pkt_data_B = cp.concatenate((cp.zeros(Config.nsamp // 2, dtype=cp.complex64), pkt_data_A,
                                         cp.zeros(Config.nsamp // 2, dtype=cp.complex64)))
            ans_list, pkt_data_C = test_work_coarse(pkt_data_B)
            # logger.warning(f'I03_5: ANS: {[x%4 for x in tocpu(ans_list)]}')
            if test_mode: logger.warning(f'I03_5: ANS: {cp_str(ans_list)}')
            payload_data = pkt_data_C[int(Config.nsamp * (Config.sfdpos + 2 + 0.25)):]
            if len(ans_list) != Config.payload_len_expected:
                logger.warning(
                    f"E03_ANS_LEN: {Config.pkt_idx=} {len(pkt_data)=} {len(pkt_data)/Config.nsamp=} {len(ans_list)=}")
            elif not test_mode:
                outpath = os.path.join(Config.dataout_path, 'part' + str(prtidx), str(out_pkt_idx))
                if not os.path.exists(outpath): os.makedirs(outpath)
                for idx, decode_ans in enumerate(list(tocpu(ans_list))):
                    data = payload_data[Config.nsamp * idx: Config.nsamp * (idx + 1)]
                    fout_path = os.path.join(outpath, f"{idx}_{round(decode_ans) % Config.n_classes}_{out_pkt_idx}_{Config.sf}.mat")
                    assert not os.path.exists(fout_path), fout_path
                    data.tofile(fout_path)
            if test_mode:
                print(time.time() - oldtime)
                oldtime = time.time()
                if Config.pkt_idx > 10: break
        assert Config.pkt_idx != 0
        Config.file_pkt_idx += Config.pkt_idx
        Config.progress_bar.reset()



def gen_pkt(cfo, sfo, to, pkt_contents):

    # their oscilliator is correct (fs), our oscilliator is fs + sfo
    # their time start from 0, our sampling start from to < 0
    assert to < 0, "Time Offset must be < 0"
    tot_t = Config.nsamp * (Config.sfdend + len(pkt_contents))
    ts_ours = 1 / (Config.fs + sfo)
    tsymb_theirs = Config.nsamp / Config.fs
    t_all = cp.arange(to, to + tot_t * ts_ours, ts_ours)

    data_pkt = cp.zeros(t_all.shape[0], dtype=cp.complex64)

    # preamble
    for symb_idx in range(Config.preamble_len):
        istart = symb_idx * Config.nsamp
        data = mychirp(t_all,
                                f0=- Config.bw / 2 + cfo,
                                f1=Config.bw / 2 + cfo,
                                t1=tsymb_theirs * (symb_idx + 1),
                                t0 = tsymb_theirs * symb_idx)
        data[t_all <= istart / Config.fs] = 0
        data[t_all > (istart + Config.nsamp) / Config.fs] = 0
        data_pkt += data
    # two codes
    for symb_idx in range(Config.preamble_len, Config.preamble_len + 2):
        istart = symb_idx * Config.nsamp
        data = mysymb(t_all,
                                f0=- Config.bw / 2 + cfo,
                                f1=Config.bw / 2 + cfo,
                                t1=tsymb_theirs * (symb_idx + 1),
                                t0 = tsymb_theirs * symb_idx,
                                symb=pkt_contents[symb_idx - Config.preamble_len])
        data[t_all <= istart / Config.fs] = 0
        data[t_all > (istart + Config.nsamp) / Config.fs] = 0
        data_pkt += data

    # SFD 1, 2
    for symb_idx in range(Config.preamble_len + 2, Config.preamble_len + 4):
        istart = symb_idx * Config.nsamp
        data = mychirp(t_all,
                                f0=Config.bw / 2 + cfo,
                                f1=- Config.bw / 2 + cfo,
                                t1=tsymb_theirs * (symb_idx + 1),
                                t0 = tsymb_theirs * symb_idx)
        data[t_all <= istart / Config.fs] = 0
        data[t_all > (istart + Config.nsamp) / Config.fs] = 0
        data_pkt += data

    # SFD 2.25
    symb_idx = Config.preamble_len + 4
    istart = symb_idx * Config.nsamp
    data = mychirp(t_all,
                            f0=Config.bw / 2 + cfo,
                            f1=Config.bw / 4 + cfo,
                            t1=tsymb_theirs * (symb_idx + 0.25),
                            t0 = tsymb_theirs * symb_idx)
    data[t_all <= istart / Config.fs] = 0
    data[t_all > (istart + Config.nsamp * 0.25) / Config.fs] = 0
    data_pkt += data

    for symb_code_idx in range(2, pkt_contents.shape[0]):
        symb_idx = symb_code_idx - 2 + Config.preamble_len + 4.25
        istart = symb_idx * Config.nsamp
        data = mysymb(t_all,
                                f0=- Config.bw / 2 + cfo,
                                f1=Config.bw / 2 + cfo,
                                t1=tsymb_theirs * (symb_idx + 1),
                                t0 = tsymb_theirs * symb_idx,
                                symb=pkt_contents[symb_code_idx])
        data[t_all <= istart / Config.fs] = 0
        data[t_all > (istart + Config.nsamp) / Config.fs] = 0
        data_pkt += data
    snr = 10
    amp = math.pow(0.1, snr / 20) * cp.mean(cp.abs(data_pkt))
    noise = (amp / math.sqrt(2) * cp.random.randn(data_pkt.shape[0])
             + 1j * amp / math.sqrt(2) * cp.random.randn(data_pkt.shape[0]))
    data_pkt = data_pkt + noise.astype(cp.complex64)  # dataX: data with noise

    return data_pkt

def test():
    to = - 2 ** Config.sf / Config.bw * 0.3
    pkt_contents = np.concatenate((np.array((16, 24), dtype=int), np.arange(0, 3000, 100, dtype=int)))
    cfo = 200
    sfo = cfo * Config.fs / Config.sig_freq
    print(sfo)
    est_cfo_slope = cfo / Config.sig_freq * Config.bw * Config.fs / Config.nsamp
    print(est_cfo_slope, sfo)
    pkt = gen_pkt(cfo = cfo, sfo = sfo, to = to, pkt_contents = pkt_contents)
    # pkt = cp.concatenate((cp.zeros(Config.nsamp, dtype=cp.complex64),
    #                       pkt,
    #                       cp.zeros(Config.nsamp, dtype=cp.complex64)))
    outpath = "."
    pkt.tofile(os.path.join(outpath, f"test.sigdat"))

    ans_list, pkt_data_C = test_work_coarse(pkt)
    print(ans_list)

if __name__ == "__main__":
    main()
    #test()


