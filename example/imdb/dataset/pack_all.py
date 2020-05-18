import numpy as np
from natsort import natsorted
from sklearn.utils.extmath import randomized_svd

if __name__ == '__main__':
    dataset = np.load('data/imdb_preprocessed.pkl', allow_pickle=True)
    validation_set = open('data/validation_set.txt').readlines()
    validation_set = set([x.strip() for x in validation_set])

    freqs = open('data/glove_output/vocab.txt').readlines()
    freqs = [(x.strip().split()[0].strip(), int(x.strip().split()[1])) \
             for x in freqs]
    dictionary = {k[0]:i+1 for i,k in enumerate(freqs)}
    inverse_dictionary = {dictionary[k]:k for k in dictionary}
    
    X_train, y_train = [], []
    X_valid, y_valid = [], []
    X_test, y_test = [], []
    X_unsup, y_unsup = [], []
    for review in natsorted(dataset.keys()):
        sentences = dataset[review].split('\n')
        sentences = [[dictionary[word] for word in sentence.split() \
                      if word in dictionary] for sentence in sentences ]
    
        sentences = [s for s in sentences if len(s) > 0]
        if len(sentences) > 0:
            label = -1
            if 'neg' in review:
                label = 0
            if 'pos' in review:
                label = 1
            if 'unsup' in review:
                X_unsup.append(sentences)
                y_unsup.append(label)
    
            elif 'test' in review:
                X_test.append(sentences)
                y_test.append(label)
            else:
                if review in validation_set:
                    X_valid.append(sentences)
                    y_valid.append(label)
                else:
                    X_train.append(sentences)
                    y_train.append(label)
    X_train, y_train = np.array(X_train), np.array(y_train).astype(np.int32)
    X_valid, y_valid = np.array(X_valid), np.array(y_valid).astype(np.int32)
    X_test, y_test   = np.array(X_test), np.array(y_test).astype(np.int32)
    X_unsup, y_unsup = np.array(X_unsup), np.array(y_unsup).astype(np.int32)

    raw_wv = open('data/glove_output/vectors.txt').readlines()
    # zero index reserved for padding
    word_vectors = np.zeros((len(dictionary)+1, len(raw_wv[0].split())-1), \
            dtype=np.float32)

    for rwv in raw_wv:
        word, vector = rwv.split()[0], np.array(rwv.split()[1:]).astype(np.float32)
        word = word.strip()
        if word in dictionary:
            word_vectors[dictionary[word]] = vector

    normalized_word_vectors = word_vectors.copy()
    normalized_word_vectors[1:] /= np.linalg.norm(normalized_word_vectors[1:], \
            axis=1, keepdims=True)

    U, Sigma, VT = randomized_svd(normalized_word_vectors[1:], 
                                  n_components=100,
                                  n_iter=20,
                                  random_state=42)
    best_v = None
    best_cos_sim = 1.0
    for V in VT:
        idx = (V*word_vectors[1:]).sum(axis=1).argsort()[::-1][0]
        cos_sim = (V*word_vectors[1:][idx]).sum()
        cos_sim = abs(cos_sim)
        if (cos_sim < best_cos_sim):
            best_cos_sim = cos_sim
            best_v = V
    normalized_word_vectors[0] = V

    imdb = {'train': {'X': X_train, 'y':y_train},
            'valid': {'X': X_valid, 'y':y_valid}, 
            'test' : {'X': X_test,  'y':y_test},
            'unsup': {'X': X_unsup, 'y':y_unsup},
            'dictionary': dictionary,
            'inverse_dictionary': inverse_dictionary,
            'freqs': freqs,
            'word_vectors': word_vectors,
            'normalized_word_vectors': normalized_word_vectors}
    np.save('data/imdb.npy', imdb)

