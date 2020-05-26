import numpy as np

def get_ngrams(sentence, n=3):
    return list(zip(*[sentence[i:] for i in range(n)])) 

class IMDB:
    def __init__(self, X, y, dictionary, inverse_dictionary, padding=True):
        self.X = X
        self.y = y
        self.dictionary = dictionary
        self.inverse_dictionary = inverse_dictionary
        self.padding = padding
        
    """
    indices: np.array of indices
    """
    def get_batch(self, indices):
        reviews = self.X[indices]
        labels  = self.y[indices]
        x_batch = []
        y_batch = []
        s_batch = []
        review_count = 0
        for review,label in zip(reviews,labels):
            x = []
            s = []
            sentence_count = 0
            for sentence in review:
                if self.padding:
                    sentence = [0] + sentence + [0]
                ngrams = get_ngrams(sentence)
                if len(ngrams) > 100:
                    ngrams = ngrams[:50] + ngrams[-50:]
                if len(ngrams) > 0:
                    x += ngrams
                    s += [[review_count, sentence_count]]*len(ngrams)
                    sentence_count += 1
            if len(x) > 0:
                x_batch += x
                s_batch += s
                y_batch += [label]
                review_count += 1
        return np.array(x_batch).astype(np.int32), \
               np.array(s_batch).astype(np.int32), \
               np.array(y_batch).astype(np.float32)[:,None]
    
    def flow(self, batch_size, drop_last=True, shuffle=True):
        def _flow():
            idx = np.arange(len(self.X))
            if drop_last:
                n_batches = len(idx) // batch_size
            else:
                n_batches = int(np.ceil(len(idx) / batch_size))
            if shuffle:
                np.random.shuffle(idx)
            for b in range(n_batches):
                li, ri = b*batch_size, min((b+1)*batch_size, len(idx))
                current_idx = idx[li:ri]
                x,s,y = self.get_batch(current_idx)
                yield x,s,y
        return _flow
