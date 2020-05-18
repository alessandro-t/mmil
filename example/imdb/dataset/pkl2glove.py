import numpy as np
from natsort import natsorted

if __name__ == '__main__':
    pklName = 'data/imdb_preprocessed.pkl'
    validSet = open('data/validation_set.txt').readlines()
    validSet = set([x.strip() for x in validSet])
    dataset = np.load(pklName, allow_pickle=True)
    glove_str = ''
    tot_sentences = {'train':[], 'test': []}
    tot_words = {'train':[], 'test': []}
    for key in natsorted(list(dataset.keys())):
        sentences = dataset[key].strip().split('\n')
        words = ' '.join(sentences).split()
        if 'train' in key:
            tot_sentences['train'].append(len(sentences))
            tot_words['train'].append(len(words))
        else:
            tot_sentences['test'].append(len(sentences))
            tot_words['test'].append(len(words))

        if 'train' in key and key not in validSet:
            glove_str += ' ' + ' '.join(words)
    
    with open('data/imdb_for_glove.txt', 'w') as glove_file:
        glove_file.write(glove_str.strip())
    print('Mean Words: {:.2f}'.format(np.mean(tot_words['train'])))
    print('Mean Sentences: {:.2f}'.format(np.mean(tot_sentences['train'])))

          
