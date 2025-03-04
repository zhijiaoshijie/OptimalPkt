from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
import plotly.express as px

from utils import *
from pltfig import pltfig1
import scipy.signal as signal
from mainwork import mainwork
import sys

def preprocess_file(file_path, outpath):
    # with open(file_path, 'rb') as file:
    #     peak = 1750 - Config.preamble_len - 2
    #     file.seek(around(max(peak, 0) * Config.nsamp * 4 * 2))
    #     rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp * (Config.total_len + 20))
    #     rawdata.tofile("test.sigdat")
    #     mainwork(0, rawdata, '/data/djl/OptimalPkt')
    # sys.exit(0)
    if  'lot-cover-10-4-E63' in file_path: return

    pkt_idx = 0
    #  read file and count size
    logger.warning(f"FILEPATH {file_path}")
    fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
    logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')
    # read max power of first 5000 windows, for envelope detection

    # power_eval_len = 5000
    power_skip_len = 15

    beta = Config.bw / ((2 ** Config.sf) / Config.bw)
    tstandard = cp.arange(Config.nsamp) / Config.fs
    refchirp = cp.exp(-1j * 2 * cp.pi * (-Config.bw * 0.5 * tstandard + 0.5 * beta * tstandard * tstandard))

    nmaxs = []
    data2s = cp.zeros((Config.preamble_len, Config.nsamp), dtype=cp.float32)
    for idx, rawdata in enumerate(read_large_file(file_path)):
        if idx < power_skip_len: nmaxs.append(0)
        else:
            data2 = rawdata * refchirp
            data3 = cp.abs(myfft(data2, Config.nsamp, Config.plan2))
            data3a = cp.convolve(data3, cp.array([1, 1, 1]), mode='same')
            data2s[idx % Config.preamble_len] = data3a
            datav = cp.sum(data2s, axis=0)
            nmaxs.append(cp.max(datav).item())
    nmaxs = tocpu(cp.array(nmaxs))
    # pltfig1(None, nmaxs, title=f"{file_path}").show()
    prominence = 0.1 if 'case' in file_path else 0.2
    # prominence = 0.2
    peaks, properties = signal.find_peaks(nmaxs, prominence=prominence, distance=Config.total_len)  # Detect peaks above height 0
    logger.warning(f"{file_path} {len(peaks)=} {peaks[0]=} {peaks[-1]=}")
    # Plot result
    # sys.exit(0)
    # pltfig1(None, peaks, title="peak positions").show()
    peaks = cp.array(peaks) - Config.preamble_len - 1 - 4 # todo!!!

    differences = np.diff(peaks)
    common_diff = np.nanmedian(differences)
    n = around((peaks[-1] - peaks[0]) / common_diff) + 1
    peaks = peaks[0] + common_diff * cp.arange(n)
    # pltfig1(None, nmaxs, addvline=peaks, title=f"{file_path}").show()
    # return

    for peak in peaks:
        with open(file_path, 'rb') as file:
            # Move the file pointer to the desired position (e.g., 100 bytes from the start)
            file.seek(around(max(peak, 0) * Config.nsamp * 4 * 2))
            rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp * (Config.total_len + 30))
            mainwork(pkt_idx, rawdata, outpath) # TODO
            try:
                mainwork(pkt_idx, rawdata, outpath)
            except Exception as e:
                logger.error(str(e))
            pkt_idx += 1
    return pkt_idx

def read_large_file(file_path_in):
    with open(file_path_in, 'rb') as file:
        # t = 1.45e6
        while True:
            try:
                rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp)
                # t-=len(rawdata)
            except EOFError:
                logger.info(f"file complete with EOF {file_path_in=}")
                break
            if len(rawdata) < Config.nsamp:
                logger.info(f"file complete{file_path_in=}, {len(rawdata)=}")
                break
            # if t<0:
            #     plt.scatter(x=np.arange(Config.nsamp - 1),
            #                 y=cp.diff(cp.unwrap(cp.angle(rawdata[:Config.nsamp]))).get(), s=0.2)
            #     plt.show()
            yield rawdata



def read_pkt(file_path_in1, threshold, min_length=15):
    current_sequence1 = []

    read_idx = -1
    for rawdata1 in read_large_file(file_path_in1):
        read_idx += 1

        number1 = cp.max(cp.abs(rawdata1))
        # if read_idx > 12564: logger.warning(f"{read_idx=} {number1=}")

        # Check for threshold in both files
        if number1 > threshold:
            current_sequence1.append(rawdata1)
        else:
            if len(current_sequence1) > min_length:
                # if read_idx > 12564:
                     #logger.warning(f"end {read_idx=} {threshold=} {number1=}")
                     # pltfig1(None, cp.unwrap(cp.angle(rawdata1)), title="read_pkt ending code").show()
                current_sequence1.append(rawdata1) # end +1 window
                yield read_idx + 1 - len(current_sequence1), cp.concatenate(current_sequence1)
            current_sequence1 = [rawdata1,] # previous +1 window

    # Yield any remaining sequences after the loop
    if len(current_sequence1) > min_length:
        yield read_idx, cp.concatenate(current_sequence1)


