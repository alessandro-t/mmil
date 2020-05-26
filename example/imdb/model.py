from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

import sys
sys.path.append('../../lib')

from mmil import BagLayer, reduce_segment_ids

class Encoder(Model):
    def __init__(self, embeddings):
        super(Encoder, self).__init__()
        self.emb = Embedding(embeddings.shape[0],
                  embeddings.shape[1],
                  weights=[embeddings],
                  input_length=3,
                  trainable=False)
    def call(self, x):
        return self.emb(x)
    
class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.reshape = Reshape((300, 1))
        self.conv1 = Conv1D(300, kernel_size=100, strides=100, \
                            activation='relu', \
                            kernel_initializer='he_normal')
        self.flatten = Flatten()

        self.d1a = Dense(250, activation='relu',\
                        kernel_initializer='he_normal')
        self.d1b = Dense(250, activation='relu',\
                         kernel_initializer='he_normal')
    
        self.bl1a = BagLayer('max')
        self.bl1b = BagLayer('mean')
        self.concat1 = Concatenate()
        
        self.d2a = Dense(250, activation='relu',\
                        kernel_initializer='he_normal')
        self.d2b = Dense(250, activation='relu',\
                         kernel_initializer='he_normal')
    
        self.bl2a = BagLayer('max')
        self.bl2b = BagLayer('mean')
        self.concat2 = Concatenate()

        self.d3 = Dense(1, activation='sigmoid',\
                        kernel_initializer='he_uniform')

        self.get_activation_before_bl = False

    def set_activation_before_bl(self, bl):
        self.get_activation_before_bl = bl
        
    def call(self, x):
        x, s = x
        out_1 = reduce_segment_ids(s, 1)
        out_2 = reduce_segment_ids(s, 2)
        
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.flatten(x)
        x1 = self.d1a(x)
        x2 = self.d1b(x)
        if self.get_activation_before_bl:
           int1 = tf.concat([x1,x2],axis=-1)

        x1 = self.bl1a([x1,out_1])
        x2 = self.bl1b([x2,out_1])
        x  = self.concat1([x1,x2])
        
        x1 = self.d2a(x)
        x2 = self.d2b(x)
        if self.get_activation_before_bl:
           int2 = tf.concat([x1,x2],axis=-1)

        x1 = self.bl2a([x1,out_2])
        x2 = self.bl2b([x2,out_2])
        x  = self.concat2([x1,x2])
        if self.get_activation_before_bl:
            return int1, int2, self.d3(x)
        else:
            return self.d3(x)
