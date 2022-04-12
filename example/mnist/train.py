from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm, trange

import numpy as np

import sys
sys.path.append('../..')
from lib.mmil import *

BATCH_SIZE = 20
# IN THE PAPER WE USED 200 EPOCHS 
# BUT 100 ARE ENOUGH
EPOCHS = 100
PATIENCE = 20

def get_model(embeddings):
    in_ = Input(shape=(1,), dtype='int32')
    seg_idx = Input(shape=(2,), name='seg_idsx', dtype='int32')
    out_1 = Lambda(lambda x: reduce_segment_ids(x, 1), output_shape=(1,))(seg_idx) 
    out_2 = Lambda(lambda x: reduce_segment_ids(x, 2), output_shape=(1,))(seg_idx) 

    x = in_
    x = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(x)
    x = Reshape((28,28,1))(x)
    x = Conv2D(32, kernel_size=(5,5), strides=2, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, kernel_size=(5,5), strides=2, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(1024, activation='linear')(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(200, name='before_bag_1')(x)
    x = ReLU()(x)
    x = BagMergeMax()([x, out_1])

    x = Dense(200, name='before_bag_2')(x)
    x = ReLU()(x)
    x = BagMergeMax()([x, out_2])
   
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model([in_,seg_idx], x)
    return model

def get_batch(data, batch_idx):
    x_batch = []
    s_batch = []
    for i,t in enumerate(data['idx'][batch_idx]):
        for j,s in enumerate(t):
            x_batch.append(s)
            s_batch.append([[i,j]]*len(s))
    return np.concatenate(x_batch)[:,None].astype(np.int32), \
           np.concatenate(s_batch).astype(np.int32), \
           data['labels'][batch_idx][:,None].astype(np.float32)

class ModelHelper:
    def __init__(self, model, optimizer, loss_object):
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object


    @tf.function(input_signature=[tf.TensorSpec(shape=(None,1), dtype=tf.int32), \
                                  tf.TensorSpec(shape=(None,2), dtype=tf.int32), \
                                  tf.TensorSpec(shape=(None,1), dtype=tf.float32)])
    def train(self,images, segs, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model([images,segs], training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,1), dtype=tf.int32), \
                                  tf.TensorSpec(shape=(None,2), dtype=tf.int32)])
    def predict(self, images, segs):
        predictions = self.model([images,segs], training=False)
        return predictions

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,1), dtype=tf.int32), \
                                  tf.TensorSpec(shape=(None,2), dtype=tf.int32), \
                                  tf.TensorSpec(shape=(None,1), dtype=tf.float32)])
    def eval(self, images, segs, labels):
        predictions = self.model([images,segs], training=False)
        loss = self.loss_object(labels, predictions)
        return loss


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.reshape((-1, 28*28)) / 255).astype(np.float32)
    X_test  = (X_test.reshape((-1, 28*28))  / 255).astype(np.float32)

    embeddings = np.vstack((X_train, X_test))
    data = np.load('data/mnist_mmil.npy', allow_pickle=True)[()]

    model = get_model(embeddings)
    optimizer = Adam()
    loss_object = BinaryCrossentropy()

    train_tb_idx = np.arange(len(data['train']['idx']))
    model_helper = ModelHelper(model, optimizer, loss_object)
    best_valid_loss = np.inf
    count = 0
    for epoch in tqdm(range(EPOCHS)):
        np.random.shuffle(train_tb_idx)
        if epoch > 80:
            K.set_value(optimizer.lr, 1e-4)
        loss = 0
        batches = trange(len(train_tb_idx)//BATCH_SIZE, desc='Train Loss: inf')
        for b,batch in enumerate(batches):
            batch_idx = train_tb_idx[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            x_batch, s_batch, y_batch = get_batch(data['train'], batch_idx)
            loss += (model_helper.train(x_batch, s_batch, y_batch)).numpy()
            batches.set_description('Train Loss: {:.3f}'.format(loss / (b+1)))
        # batches.reset()

        # COMPUTE LOSS ON VALIDATION
        x_batch, s_batch, y_batch = get_batch(data['valid'], np.arange(len(data['valid']['idx'])))
        valid_loss = (model_helper.eval(x_batch, s_batch, y_batch)).numpy()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            count = 0
            model.save('model.h5')
        else:
            count = count + 1

        if count == PATIENCE:
            print('Early Stopping')
            break

    # EVAL:
    model.load_weights('model.h5')
    x_batch, s_batch, y_batch = get_batch(data['test'], np.arange(len(data['test']['idx']))) 
    x_batch += len(X_train)
    preds = (model_helper.predict(x_batch, s_batch)).numpy() > 0.5
    accuracy = 100*np.mean(preds == y_batch)
    print('Test Accuracy: {:.3f}'.format(accuracy))






