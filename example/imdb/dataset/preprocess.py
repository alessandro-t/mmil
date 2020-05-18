import numpy as np
import os
import pickle                                                                   
import subprocess
import sys
from concurrent import futures
from natsort import natsorted
                                                                                
def save_object(obj, filename):                                                 
    with open(filename, 'wb') as output:  # Overwrites any existing file.       
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)                       
                                                                                
def load_object(filename):                                                      
    with open(filename, 'rb') as input:                                         
        return pickle.load(input)   

def _parallel_exec(args):
    data = {}
    files = args['chunk']
    for file in files:
        result = subprocess.run(['java', '-cp', nlpPath, \
                     'edu.stanford.nlp.process.DocumentPreprocessor', file],\
                     stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8')
        data[file] = result
    return data

if __name__ == '__main__':
    processes  = int(sys.argv[1])
    imdbPath   = 'data/aclImdb/'
    nlpPrefix  = '../../../tools'
    nlpFolder  = [folder for folder in os.listdir(nlpPrefix) \
                  if folder.startswith('stanford-corenlp')]
    nlpPath = os.path.join(nlpPrefix, nlpFolder[0], nlpFolder[0] + '.jar')

    data = {'train': ['pos', 'neg', 'unsup'], 'test':['pos', 'neg']}
    
    all_files = []
    for set_ in data:
        for k in data[set_]:
            fpath = os.path.join(imdbPath, set_, k)
            for fname in natsorted(os.listdir(fpath)):
                if fname.endswith('.txt'):
                    file_path = os.path.join(fpath, fname)
                    all_files.append(file_path)

    chunks = np.array_split(all_files, processes)
    data = [{'chunk':chunk} for chunk in chunks]
    pool = futures.ProcessPoolExecutor(max_workers=processes)
    result = list(pool.map(_parallel_exec, data))
    pool.shutdown()
    
    dataset = {}
    for res in result:
        dataset.update(res)
       
    save_object(dataset, 'data/imdb_preprocessed.pkl')

