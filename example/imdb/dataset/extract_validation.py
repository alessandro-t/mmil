import numpy as np
import sys

from natsort import natsorted

if __name__ == '__main__':
    pkl_data = np.load('data/imdb_preprocessed.pkl', allow_pickle=True)
    keys = natsorted([k for k in pkl_data.keys() if 'train/pos' in k or \
                      'train/neg' in k])
    np.random.seed(int(sys.argv[2]))
    valid_size = int(np.round(len(keys) * float(sys.argv[1])))
    valid_keys = np.random.choice(keys, valid_size, replace=False)
    valid_keys = natsorted(valid_keys.tolist())
    with open('data/validation_set.txt', 'w') as valid_file:
        valid_file.write('\n'.join(valid_keys))
