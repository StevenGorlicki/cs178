import pickle
import numpy as np
from typing import List, Tuple

def unpickle_f(files: List[str]) -> Tuple[List[dict], np.ndarray, np.ndarray]:
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
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return dicts, xs, ys
