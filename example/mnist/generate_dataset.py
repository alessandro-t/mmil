import numpy as np
import os
from tensorflow.keras.datasets import mnist

# RANDOM SEED IS SET FOR REPRODUCIBILITY
SEED = 43
np.random.seed(SEED)

def create_dataset(idx, labels, size=5000, max_top_bags=10, max_sub_bags=10):
    pos = size // 2
    neg = size // 2
    pos_found = 0
    neg_found = 0
    top_bags = []
    top_bags_labels = []
    while len(top_bags) < (pos + neg):
        n_top_bags = np.random.randint(2, max_top_bags+1)
        all_index = []
        is_positive = False
        for t in range(n_top_bags):
            sub_bag_labels = np.random.randint(0, 10, np.random.randint(2, \
                                               max_sub_bags+1))
            sub_bag_index = []
            for label in sub_bag_labels:
                sub_bag_index += [idx[np.random.choice(np.where(labels[idx] == label)[0],1)]]
            sub_bag_index = np.array(sub_bag_index).ravel()
            if (7 in set(labels[sub_bag_index])) and \
               (3 not in set(labels[sub_bag_index])):
                is_positive = True
            all_index += [sub_bag_index]
            
        if is_positive and pos_found < pos:
            top_bags += [np.array(all_index)]
            top_bags_labels += [1]
            pos_found += 1
            
        if not is_positive and neg_found < neg:
            top_bags += [np.array(all_index)]
            top_bags_labels += [0]
            neg_found += 1

    tb_index  = np.array(top_bags)
    tb_labels = np.array(top_bags_labels)
    return tb_index, tb_labels

if __name__ == '__main__':
    train_size = 5000
    valid_size = 1000
    test_size  = 5000

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    train_idx = np.sort(np.random.choice(np.arange(len(X_train)), \
                        int(len(X_train)*0.8), replace=False))
    valid_idx = np.sort(np.array(list(set(np.arange(len(X_train))) - \
                                 set(train_idx))))
    test_idx  = np.arange(len(X_test))

    train_tb_index, train_tb_labels = create_dataset(train_idx, y_train, \
                                                     size=train_size)
    valid_tb_index, valid_tb_labels = create_dataset(valid_idx, y_train, \
                                                     size=valid_size)
    test_tb_index,  test_tb_labels  = create_dataset(test_idx,  y_test,  \
                                                     size=valid_size)

    os.makedirs('data', exist_ok=True)
    data = {
            'train': {'idx': train_tb_index, 'labels':train_tb_labels},
            'valid': {'idx': valid_tb_index, 'labels':valid_tb_labels},
            'test':  {'idx': test_tb_index,  'labels':test_tb_labels}
           }
    np.save('data/mnist_mmil.npy', data)
