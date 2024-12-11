import itertools

from utils import *


def read_large_file(file_path_in):
    with open(file_path_in, 'rb') as file:
        # t = 1.45e6
        while True:
            try:
                rawdata = cp.fromfile(file, dtype=cp.complex64, count=Config.nsamp)
                # t-=len(rawdata)
            except EOFError:
                logger.warning(f"file complete with EOF {file_path_in=}")
                break
            if len(rawdata) < Config.nsamp:
                logger.warning(f"file complete{file_path_in=}, {len(rawdata)=}")
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
                current_sequence1.append(rawdata1)
                current_sequence2.append(rawdata2)
                yield read_idx, cp.concatenate(current_sequence1), cp.concatenate(current_sequence2)
            current_sequence1 = []
            current_sequence2 = []

    # Yield any remaining sequences after the loop
    if len(current_sequence1) > min_length:
        yield read_idx, cp.concatenate(current_sequence1), cp.concatenate(current_sequence2)


