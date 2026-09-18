"""Customer-side converter: MANO_RIGHT.pkl -> mano_right.npz (plain tensors).

Customers register at https://mano.is.tue.mpg.de, download MANO_RIGHT.pkl, and
run this script locally once. The resulting .npz contains only plain numpy
arrays (no chumpy/pickle objects), which:
  - is what the 3DHandReconstruction SageMaker endpoint accepts in the inference request
    (base64-encoded, cached server-side by checksum),
  - loads in milliseconds without the abandoned `chumpy` dependency.

Requires (only for this script, not for using the endpoint):
    pip install "numpy<1.24" chumpy scipy

Usage:
    python mano_to_npz.py MANO_RIGHT.pkl mano_right.npz
"""
import argparse
import pickle
import sys

import numpy as np

# Arrays smplx's MANO layer consumes. scipy sparse J_regressor is densified.
INT_KEYS = ['f', 'kintree_table']
KEYS = ['v_template', 'shapedirs', 'posedirs', 'J_regressor', 'weights',
        'hands_components', 'hands_mean', 'hands_coeffs'] + INT_KEYS
REQUIRED = ['v_template', 'shapedirs', 'posedirs', 'J_regressor', 'weights',
            'f', 'kintree_table', 'hands_components', 'hands_mean']


def to_array(value):
    # chumpy arrays, scipy sparse matrices, and plain ndarrays all normalize
    # through np.asarray / .toarray
    if hasattr(value, 'toarray'):
        return np.asarray(value.toarray())
    return np.asarray(value)


def convert_pkl(pkl_path):
    """Load a MANO pkl and return a dict of plain numpy arrays."""
    try:
        import chumpy  # noqa: F401  (needed so pickle can resolve ch.Ch objects)
    except ImportError:
        raise ImportError('chumpy is required to read the original pkl: '
                          'pip install "numpy<1.24" chumpy scipy')

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    out = {}
    for key in KEYS:
        if key in data:
            arr = to_array(data[key])
            out[key] = arr.astype(np.int64 if key in INT_KEYS else np.float64)
    missing = [k for k in REQUIRED if k not in out]
    if missing:
        raise ValueError(f'Input does not look like a MANO model file; '
                         f'missing keys: {missing}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('pkl_in')
    parser.add_argument('npz_out')
    args = parser.parse_args()

    try:
        out = convert_pkl(args.pkl_in)
    except (ImportError, ValueError) as exc:
        sys.exit(str(exc))

    np.savez(args.npz_out, **out)
    sizes = {k: list(v.shape) for k, v in out.items()}
    print(f'Wrote {args.npz_out}: {sizes}')


if __name__ == '__main__':
    main()
