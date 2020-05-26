from functools import partial
from tensorflow.keras.optimizers import Adam
from time import time
from tqdm.auto import tqdm

import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K

from model import Encoder, Decoder
from dataset import IMDB

EPSILON = 2.200

class TrainHelper:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = Adam()
        self.loss_object = tf.losses.binary_crossentropy

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,3), dtype=tf.int32), \
                                 tf.TensorSpec(shape=(None,2), dtype=tf.int32), \
                                 tf.TensorSpec(shape=(None,1), dtype=tf.float32)])
    def train(self, x, segments, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.decoder([self.encoder(x), segments], training=True)
            mask = tf.cast(tf.greater_equal(labels, -0.5), tf.float32)                 
            loss = self.loss_object(labels, predictions) * tf.squeeze(mask,-1)
            num_samples = tf.math.maximum(tf.reduce_sum(tf.squeeze(mask,-1)), 1.0)                 
            loss = tf.reduce_sum(loss) / num_samples       
            
            x_dense = self.encoder(x)
            vae_loss = self.virtual_adversarial_loss(partial(self.decoder, training=True), x_dense , segments)
            total_loss = loss + vae_loss
            
        gradients = tape.gradient(total_loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))
        return total_loss

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,3), dtype=tf.int32), \
                                 tf.TensorSpec(shape=(None,2), dtype=tf.int32)])
    def predict(self, x, segments):
        predictions = self.decoder([self.encoder(x), segments], training=False)
        return predictions

    def kl_divergence(self, q, p):
        q = tf.clip_by_value(q, 1e-7, 1.0-1e-7)
        p = tf.clip_by_value(p, 1e-7, 1.0-1e-7)
        qlogq = -tf.losses.binary_crossentropy(q, q)
        qlogp = -tf.losses.binary_crossentropy(q, p)
        return tf.reduce_mean(qlogq - qlogp)

    def get_normalized_vector(self, x):
        shape = K.shape(x)
        alpha = K.max(K.abs(x), -1, keepdims=True) + 1e-12
        l2_norm = alpha * K.sqrt(
            K.sum(K.pow(x / alpha, 2), -1, keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return K.reshape(x_unit, shape)
    
    def virtual_adversarial_loss(self, f, x, segments):
        r_vadv = self.generate_virtual_adversarial_perturbation(f, x, segments)
        logit = tf.stop_gradient(f([x, segments]))
        logit_p = logit
        logit_m = f([x + r_vadv, segments])
        loss = self.kl_divergence(logit_p, logit_m)
        return loss
    
    def generate_virtual_adversarial_perturbation(self, f, x, segments):
        xi = 1e-1
        num_power_iterations = 1
        epsilon = EPSILON
        
        logit = f([x, segments])
        
        d = tf.random.normal(shape=tf.shape(x))
        for _ in range(num_power_iterations):
            d = xi * self.get_normalized_vector(d)
            logit_p = logit
            with tf.GradientTape() as t:
                t.watch(d)
                x_d = x + d
                logit_m = f([x_d, segments])
                dist = self.kl_divergence(logit_p, logit_m)
            grad = t.gradient(dist, [d])[0]
            grad = tf.stop_gradient(grad)
        return epsilon * self.get_normalized_vector(grad)
            
if __name__ == '__main__':
    imdb = np.load('dataset/data/imdb.npy', allow_pickle=True)[()]
    train = IMDB(np.concatenate((imdb['train']['X'],imdb['unsup']['X'])), \
                 np.concatenate((imdb['train']['y'],imdb['unsup']['y'])), \
                 imdb['dictionary'], imdb['inverse_dictionary'])
    valid = IMDB(imdb['valid']['X'], imdb['valid']['y'], \
                 imdb['dictionary'], imdb['inverse_dictionary'])
    test  = IMDB(imdb['test']['X'], imdb['test']['y'], \
                 imdb['dictionary'], imdb['inverse_dictionary'])
    enc = Encoder(imdb['word_vectors'])
    dec = Decoder()
    th = TrainHelper(enc, dec)
    
    batch_size = 128
    train_gen = tf.data.Dataset.from_generator(train.flow(batch_size, drop_last=True, shuffle=True),\
                                                         (tf.int32, tf.int32, tf.float32),\
                                                         (tf.TensorShape([None, 3]), \
                                                          tf.TensorShape([None, 2]), \
                                                          tf.TensorShape([None, 1])))
    test_gen = tf.data.Dataset.from_generator(test.flow(batch_size, drop_last=False, shuffle=False),\
                                                       (tf.int32, tf.int32, tf.float32),\
                                                       (tf.TensorShape([None, 3]), \
                                                        tf.TensorShape([None, 2]), \
                                                        tf.TensorShape([None, 1])))

    train_batches = len(train.X) // batch_size
    test_batches  = int(np.ceil(len(test.X) / batch_size))

    epochs = tqdm(range(20), desc='Loss: inf')
    for epoch in epochs:
        loss = []
        count = 0
        for x,s,y in tqdm(train_gen, total=train_batches, leave=False):
            loss  += [len(y) * th.train(x,s,y)]
            count += len(y)
        epochs.set_description('Loss: {:.3f}'.format(np.sum(loss)/count))
        if epoch == 6:
            K.set_value(th.optimizer.learning_rate, 1e-4)
        if epoch == 9:
            K.set_value(th.optimizer.learning_rate, 1e-5)
        # print('Train Loss: {:.3f}'.format(np.sum(loss) / count))

    acc_test = 0
    for x,s,y in tqdm(test_gen, total=test_batches, leave=False):
        y = y.numpy().ravel()
        y_pred = ((th.predict(x,s) > 0.5).numpy()).ravel()
        acc_test += np.sum(y_pred == y)
    acc_test = 100*acc_test / len(test.X)
    print('Test accuracy (EPS: {:.2f}): {:.3f}\n'.format(EPSILON, acc_test) + '-'*21)
    if not os.path.exists('models'):
        os.makedirs('models')
    signature = str(int(time()))
    enc.save_weights('models/encoder_{}.h5'.format(signature))
    dec.save_weights('models/decoder_{}.h5'.format(signature))
