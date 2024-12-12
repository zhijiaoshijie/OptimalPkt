import itertools
from sklearn.mixture import GaussianMixture

from utils import *



def preprocess_file(file_path):
    #  read file and count size
    logger.info(f"FILEPATH {file_path}")
    pkt_cnt = 0
    pktdata = []
    fsize = int(os.stat(file_path).st_size / (Config.nsamp * 4 * 2))
    logger.debug(f'reading file: {file_path} SF: {Config.sf} pkts in file: {fsize}')
    # read max power of first 5000 windows, for envelope detection
    power_eval_len = 5000
    nmaxs = []
    for idx, rawdata in enumerate(read_large_file(file_path)):
        nmaxs.append(cp.max(cp.abs(rawdata)))
        if idx == power_eval_len - 1: break
    nmaxs = cp.array(nmaxs).get()
    # clustering
    data = nmaxs.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(data)
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()
    sorted_indices = np.argsort(means)
    mean1, mean2 = means[sorted_indices]
    covariance1, covariance2 = covariances[sorted_indices]
    weight1, weight2 = weights[sorted_indices]
    # threshold to divide the noise power from signal power
    thresh = (mean1 * covariance2 + mean2 * covariance1) / (covariance1 + covariance2)
    # if threshold may not work set this to True
    # plot the power map
    if False:
        counts, bins = cp.histogram(nmaxs, bins=100)
        # logger.debug(f"Init file find cluster: counts={cp_str(counts, precision=2, suppress_small=True)}, bins={cp_str(bins, precision=4, suppress_small=True)}, {kmeans.cluster_centers_=}, {thresh=}")
        logger.debug(f"cluster: {kmeans.cluster_centers_[0]} {kmeans.cluster_centers_[1]} {thresh=}")
        threshpos = np.searchsorted(tocpu(bins), thresh).item()
        logger.debug(f"lower: {cp_str(counts[:threshpos])}")
        logger.debug(f"higher: {cp_str(counts[threshpos:])}")
        fig = px.line(nmaxs.get())
        fig.add_hline(y=thresh)
        fig.update_layout(
            title=f"{file_path} pow {len(nmaxs)}",
            legend=dict(x=0.1, y=1.1))
        fig.show()
    return thresh

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



def read_pkt(file_path_in1, file_path_in2, threshold, min_length=20):
    current_sequence1 = []
    current_sequence2 = []

    read_idx = -1
    for rawdata1, rawdata2 in itertools.zip_longest(read_large_file(file_path_in1), read_large_file(file_path_in2)):
        if rawdata1 is None or rawdata2 is None:
            break  # Both files are done
        read_idx += 1

        number1 = cp.max(cp.abs(rawdata1)) if rawdata1 is not None else 0

        # Check for threshold in both files
        if number1 > threshold:
            current_sequence1.append(rawdata1)
            current_sequence2.append(rawdata2)
        else:
            if len(current_sequence1) > min_length:
                current_sequence1.append(rawdata1) # end +1 window
                current_sequence2.append(rawdata2)
                yield read_idx, cp.concatenate(current_sequence1), cp.concatenate(current_sequence2)
            current_sequence1 = [rawdata1,] # previous +1 window
            current_sequence2 = [rawdata2,]

    # Yield any remaining sequences after the loop
    if len(current_sequence1) > min_length:
        yield read_idx, cp.concatenate(current_sequence1), cp.concatenate(current_sequence2)


