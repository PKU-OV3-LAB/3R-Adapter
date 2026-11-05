import pickle
import numpy as np
import sys
from collections import OrderedDict
if not hasattr(np, 'bool'):
    np.bool = bool
    np.int = int
    np.float = float
    np.object = object

def to_numpy(x):
    """
    将 chumpy / numpy / list / scalar 统一转为 numpy array
    """
    # chumpy variable: x.r is numpy
    if hasattr(x, 'r'):
        return np.array(x.r)

    # some is matrix-like
    if hasattr(x, 'A'):
        return np.array(x.A)

    # already numpy
    if isinstance(x, np.ndarray):
        return x

    # list or tuple
    if isinstance(x, (list, tuple)):
        return np.array(x)

    # scalar
    return x


def convert_smpl_pkl(in_pkl, out_pkl):
    print(f'Loading: {in_pkl}')
    with open(in_pkl, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    new_data = OrderedDict()

    for k, v in data.items():
        try:
            new_data[k] = to_numpy(v)
        except Exception as e:
            print(f'[WARN] key "{k}" failed to convert: {e}')
            new_data[k] = v

    print(f'Saving numpy-only SMPL to: {out_pkl}')
    with open(out_pkl, 'wb') as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # convert the chumpy structures in official SMPL files to numpy arrays
    convert_smpl_pkl('SMPL_NEUTRAL.pkl','SMPL_NEUTRAL.pkl')
    convert_smpl_pkl('SMPL_MALE.pkl','SMPL_MALE.pkl')
    convert_smpl_pkl('SMPL_FEMALE.pkl','SMPL_FEMALE.pkl')
