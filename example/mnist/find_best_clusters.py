from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import *
from tqdm import tqdm

import numpy as np
import os

import sys
sys.path.append('../..')
from lib.mmil import *

from train import get_batch, ModelHelper


BATCH_SIZE = 1
MAX_S = 10
MAX_I = 12
if __name__ == '__main__':
    os.makedirs('cluster', exist_ok=True)
    data = np.load('data/mnist_mmil.npy', allow_pickle=True)[()]

    model = load_model('model.h5', custom_objects={'BagMergeMax': BagMergeMax})
    outputs = [ model.get_layer('before_bag_1').output,\
                model.get_layer('before_bag_2').output,\
                model.output]
    model_intermediate = Model(model.inputs, outputs)
    model_helper = ModelHelper(model_intermediate, None, None)

    activations = {'train': {'lv1':[], 'lv2':[], 'output':[], 'segs':[]},
                   'valid': {'lv1':[], 'lv2':[], 'output':[], 'segs':[]},
                   'test':  {'lv1':[], 'lv2':[], 'output':[], 'segs':[]}}
    for _set in ['train', 'valid', 'test']:
        idx = np.arange(len(data[_set]['idx']))
        n_batches = np.ceil(len(idx)/BATCH_SIZE).astype(np.int32)
        for b in range(n_batches):
            l_, r_ = b*BATCH_SIZE, min( (b+1)*BATCH_SIZE, len(idx))
            x_batch, s_batch, y_batch = get_batch(data[_set], idx[l_:r_])
            if _set == 'test':
                x_batch += 60000
            preds = model_helper.predict(x_batch, s_batch)
            preds = [p.numpy() for p in preds]
            segs  = np.concatenate((np.where(np.diff(s_batch[:,1]))[0] + 1, [len(s_batch)]))
            activations[_set]['lv1'].append(preds[0])
            activations[_set]['lv2'].append(preds[1])
            activations[_set]['output'].append(preds[2])
            activations[_set]['segs'].append(segs)
            
        activations[_set]['output'] = np.vstack(activations[_set]['output']) 
    
#     print('Sub-bag clusters...')
#     for i in tqdm(range(2, MAX_S+1)):
#         clf = KMeans(n_clusters=i)
#         clf.fit(np.vstack(activations[_set]['lv2']))
#         dump(clf, 'cluster/clf_subbag_{:d}.joblib'.format(i))
#     print('Instance clusters...')  
#     for i in tqdm(range(2, MAX_I+1)):
#         clf = KMeans(n_clusters=i)
#         clf.fit(np.vstack(activations[_set]['lv1']))
#         dump(clf, 'cluster/clf_instance_{:d}.joblib'.format(i))
        
    for i in tqdm(range(2, MAX_S+1)):
        clf_s = load('cluster/clf_subbag_{:d}.joblib'.format(i))
        subbag_clusters = [clf_s.predict(k) for k in activations['valid']['lv2']]
        subbag_clusters = np.concatenate(subbag_clusters)
        for j in tqdm(range(2, MAX_I+1)):
            clf_i = load('cluster/clf_instance_{:d}.joblib'.format(j))
            
            inst_counts = []
            
            for z,k in enumerate(activations['valid']['lv1']):
                for v in range(len(activations['valid']['segs'][i])-1):
                    insts = k[activations['valid']['segs'][z][v]:activations['valid']['segs'][z][v+1]]
                    inst_counts += [np.bincount(clf_i.predict(insts), minlength=j) > 0]
 
            dt_inst_to_sb = DecisionTreeClassifier(max_depth=5, random_state=1000)
            dt_inst_to_sb = dt_inst_to_sb.fit(inst_counts, subbag_clusters)
        


