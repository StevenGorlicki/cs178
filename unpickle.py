
import numpy as np

def unpickle_f(files):
    import pickle
    xs = []
    ys = []
    dicts = []
    for file in files:

        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        X = dict[b'data']
        Y = dict[b'labels']
        xs.append(X)
        ys.append(Y)
        dicts.append(dict)
    return dicts, xs, ys