import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from functools import partial
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import sys
sys.path.append('../../lib')

from mmil import BagLayer, reduce_segment_ids

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
               np.array(y_batch).astype(np.int32), \
               np.array(s_batch).astype(np.int32)
            
def build_mmil_model(input_shape, embeddings):
    input_layer = Input(shape=input_shape)
    decoder_input = Input(shape=(input_shape[0], embeddings.shape[1]))
    segs = Input(shape=(2,), dtype='int32')
    out_1 = Lambda(lambda x: reduce_segment_ids(x, 1), \
                   output_shape=(1,))(segs)
    out_2 = Lambda(lambda x: reduce_segment_ids(x, 2), \
                   output_shape=(1,))(segs)

    x = Embedding(embeddings.shape[0],
                  embeddings.shape[1],
                  weights=[embeddings],
                  input_length=input_shape[0],
                  trainable=False)(input_layer)
    encoded = x
    encoder = Model(inputs=input_layer, outputs=encoded)

    x = decoder_input
    x = Lambda(lambda t:tf.expand_dims(t,-1))(x)
    x = Conv2D(300, kernel_size=(1,100), strides=1, activation='relu',\
               kernel_initializer='he_normal')(x)
    x = Flatten()(x)

    x1  = Dense(250, activation='relu',kernel_initializer='he_normal')(x)
    x2  = Dense(250, activation='relu',kernel_initializer='he_normal')(x)
    x1  = BagLayer('max')([x1, out_1])
    x2  = BagLayer('mean')([x2, out_1])
    x   = Concatenate()([x1, x2])

    x1  = Dense(250, activation='relu',kernel_initializer='he_normal')(x)
    x2  = Dense(250, activation='relu',kernel_initializer='he_normal')(x)
    x1  = BagLayer('max')([x1, out_2])
    x2  = BagLayer('mean')([x2, out_2])
    x   = Concatenate()([x1, x2])
            
    x = Dense(1, kernel_initializer='he_uniform')(x)

    decoded = x
    decoder = Model(inputs=[decoder_input, segs], outputs=decoded)
    model = Model(inputs=[input_layer, segs], \
                  outputs=decoder([encoder(input_layer), segs]))
    return model, encoder, decoder            

def bce(y_true, y_pred):
    return tf.losses.binary_crossentropy(y_true, y_pred, from_logits=True)

def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.sigmoid(q_logit)
    q_logit = tf.clip_by_value(q_logit, 1e-10, 1-1e-10)
    p_logit = tf.clip_by_value(p_logit, 1e-10, 1-1e-10)
    
    qlogq = tf.reduce_mean(tf.reduce_sum(q * tf.math.log( q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * tf.math.log(p_logit), 1))
    return qlogq - qlogp

class TrainHelper:
    def __init__(self, model, encoder, decoder):
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = Adam()
        self.loss_object = bce

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,3), dtype=tf.int32), \
                                  tf.TensorSpec(shape=(None,2), dtype=tf.int32), \
                                  tf.TensorSpec(shape=(None,1), dtype=tf.int32)])
#     @tf.function
    def train(self, x, segments, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model([x, segments], training=True)
            mask = tf.cast(tf.greater_equal(labels, 0), tf.float32)                 
            loss = bce(labels, predictions) * tf.squeeze(mask,-1)
            num_samples = tf.math.maximum(tf.reduce_sum(tf.squeeze(mask,-1)), 1.0)                 
            loss = tf.reduce_sum(loss) / num_samples       
            
            x_dense = self.encoder(x)
            vae_loss = self.virtual_adversarial_loss(partial(self.decoder, training=True), x_dense , segments)
            total_loss = loss + vae_loss
            
        gradients = tape.gradient(total_loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))
        return loss
    
    def virtual_adversarial_loss(self, f, x, segments):
        r_vadv = self.generate_virtual_adversarial_perturbation(f, x, segments)
        logit_p = f([x, segments])
        logit_m = f([x + r_vadv, segments])
        loss = kl_divergence_with_logit(logit_p, logit_m)
        return loss
        
    def get_normalized_vector(self, d):
        d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keepdims=True))
        d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keepdims=True))
        return d
        
    def generate_virtual_adversarial_perturbation(self, f, x, segments):
        xi = 1e-1
        num_power_iterations = 1
        epsilon = 3.0
        d = tf.random.normal(shape=tf.shape(x))

        for _ in range(num_power_iterations):
            d = xi * self.get_normalized_vector(d)
            logit_p = f([x, segments])
                   
            with tf.GradientTape() as t:
                t.watch(d)
                x_d = x + d
                logit_m = f([x_d, segments])
                dist = kl_divergence_with_logit(logit_p, logit_m)
            grad = t.gradient(dist, [d])[0]

        return epsilon * self.get_normalized_vector(grad)


if __name__ == '__main__':
    imdb = np.load('dataset/data/imdb.npy', allow_pickle=True)[()]
    train = IMDB(np.concatenate((imdb['train']['X'],imdb['unsup']['X'])), \
                 np.concatenate((imdb['train']['y'],imdb['unsup']['y'])), \
                 imdb['dictionary'], imdb['inverse_dictionary'])
    valid = imdb(imdb['valid']['x'], imdb['valid']['y'], \
                 imdb['dictionary'], imdb['inverse_dictionary'])
    test  = imdb(imdb['test']['x'], imdb['test']['y'], \
                 imdb['dictionary'], imdb['inverse_dictionary'])
    model, enc, dec = build_mmil_model((3,), imdb['normalized_word_vectors'])

    th = TrainHelper(model, enc, dec)

    train_idx = np.arange(len(train.X))
    test_idx = np.arange(len(test.X))
    batch_size = 128
    for epoch in range(20):
        acc = 0
        np.random.shuffle(train_idx)
        loss = []
        for b in range(len(train_idx) // batch_size):
            current_idx = train_idx[b*batch_size:(b+1)*batch_size]
            x,y,s = train.get_batch(current_idx)
            loss += [len(y) * th.train(x,s,y)]
            acc += batch_size
        
        for b in range(np.ceil(len(test_idx) / batch_size).astype(np.int32)):
            current_idx = test_idx[b*batch_size:(b+1)*batch_size]
            x,y,s = test.get_batch(current_idx)
            y = y.ravel()
            y_pred = (model.predict_on_batch([x,s]) > 0).ravel()
            acc_test = np.mean(y_pred == y)

        if epoch == 6:
            K.set_value(model.optimizer.lr, 1e-4)
        if epoch == 8:
            K.set_value(model.optimizer.lr, 1e-5)
        print('Loss: %.2f' % (sum(loss)/acc))
        print('Train accuracy: %.2f\nTest  accuracy: %.2f\n' % (acc_train, acc_test) + '-'*21)
        #if not os.path.exists('models'):
        #    os.makedirs('models')
        #if (epoch+1)%20 == 0:
        #model.save('models/a%s-epoch_%d-acc_%.3f' % (t_name, epoch, acc_test))
