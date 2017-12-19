import struct
import numpy as np

filenames = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def read_data(directory):
    sets = ()
    for name in filenames:
        sets = sets + ( read_idx(directory + "/" + name), )

    return (sets[0], sets[1]), (sets[2], sets[3])


