from __future__ import absolute_import, division, print_function
from operator import truediv   
from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Embedding, Dense, Dropout, Activation
from keras.engine.topology import Layer 
from keras import backend as K  
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential 
import tensorflow as tf
import numpy as np
import time 

import init
from data_generator import *
from utils import *

import sys
dataset = sys.argv[1]


if dataset == 'imdb':
    config = get_config('./config/WordGRU.json')
    batch_size = 40

elif dataset == 'agnews':
    config = get_config('./config/WordCNN.json')
    batch_size = 40

elif dataset == 'hatex':
    config = get_config('./config/WordTF.json')
    batch_size = 50


# Set parameters:
embedding_dims = 50
filters = 250


kernel_size = 3
hidden_dims = 250
epochs = 5
k = 10 #==============================
print(f'========== K = {k} ============== ')

PART_SIZE = 125


# Define various Keras layers.
Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k), output_shape=lambda x: [x[0],x[2]]) 

class Concatenate(Layer):
    def __init__(self, **kwargs): 
        super(Concatenate, self).__init__(**kwargs)
    
    def call(self, inputs):
        input1, input2 = inputs  
        input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        return tf.concat([input1, input2], axis = -1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)

class Sample_Concrete(Layer):
    def __init__(self, tau0, k, **kwargs): 
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):   
        # logits: [batch_size, d, 1]
        logits_ = K.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]

        d = int(logits_.get_shape()[2])		
        unif_shape = [batch_size,self.k,d]

        uniform = K.random_uniform_variable(shape=unif_shape,
            low = np.finfo(tf.float32.as_numpy_dtype).tiny,
            high = 1.0)
        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis = 1) 
        logits = tf.reshape(logits,[-1, d]) 
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
        
        output = K.in_train_phase(samples, discrete_logits) 
        return tf.expand_dims(output,-1)

    def compute_output_shape(self, input_shape):
        return input_shape

def construct_gumbel_selector(X_ph, num_words, embedding_dims, maxlen):
    """
    Build the L2X model for selecting words. 
    """
    emb_layer = Embedding(num_words, embedding_dims, input_length = maxlen, name = 'emb_gumbel')
    emb = emb_layer(X_ph) #(400, 50) 
    net = Dropout(0.2, name = 'dropout_gumbel')(emb)
    net = emb
    first_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(net)    

    # global info
    net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer)
    global_info = Dense(100, name = 'new_dense_1', activation='relu')(net_new) 

    # local info
    net = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(first_layer) 
    local_info = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(net)  
    combined = Concatenate()([global_info,local_info]) 
    net = Dropout(0.2, name = 'new_dropout_2')(combined)
    net = Conv1D(100, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)   

    logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net)  

    return logits_T


def L2X(train, dg, model_path, batch_size): 
    """
    Generate scores on features on validation by L2X.
    Train the L2X model with variational approaches 
    if train = True. 
    """
    max_features = dg.tokenizer.get_vocab_size()
    maxlen = dg.max_length
    

    print('Creating model...')

    # P(S|X)
    with tf.variable_scope('selection_model'):
        X_ph = Input(shape=(maxlen,), dtype='int32')

        logits_T = construct_gumbel_selector(X_ph, max_features, embedding_dims, maxlen)
        tau = 0.5
        T = Sample_Concrete(tau, k)(logits_T)

    # q(X_S)
    with tf.variable_scope('prediction_model'):
        
        emb2 = Embedding(max_features, embedding_dims, input_length=maxlen)(X_ph)
        mask = Lambda(lambda x: K.concatenate([x]*embedding_dims, axis=2))(T)

        net = Mean(Multiply()([emb2, mask]))
        net = Dense(hidden_dims)(net)
        net = Activation('relu')(net) 
        preds = Dense(dg.C, activation='softmax', name = 'new_dense')(net)


    model = Model(inputs=X_ph, outputs=preds)

    model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',#optimizer,
                    metrics=['acc']) 

    if train:
        checkpoint = ModelCheckpoint(model_path, monitor='val_acc', 
            verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint] 
        st = time.time()


        model.fit(dg.train_x, dg.train_y, validation_data=(dg.val_x, dg.val_y), callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)
        duration = time.time() - st
        print('Training time is {}'.format(duration))		
    else:
        model.load_weights(model_path, by_name=True) 

    pred_model = Model(X_ph, logits_T) 

    if dataset == 'hatex':
        batch_size = 1
    scores = pred_model.predict(dg.test_x, verbose = 1, batch_size = batch_size)

    preds = model.predict(dg.test_x, verbose = 1, batch_size = batch_size)
    scores = scores[:,:,0] 
    scores = np.reshape(scores, [scores.shape[0], maxlen])
    return preds, scores 


def validation(scores, preds, file, k):    
    iter = range(len(scores))
    for i in tqdm(iter):
        x = dg.test_x[i,:]
        text = dg.test_text[i]
        # Get label
        y_hat = np.argmax(preds[i])
        y = np.argmax(dg.test_label[i])
        # Get features
        score = scores[i, :]
        score = torch.tensor(score)
        selected = torch.topk(score, k).indices
        tokens = dg.tokenizer.decode(x[selected].tolist())
        features.append(tokens)
        content = f"{i}. {text}\n\nFeatures: {tokens}\n\nPrediction: {y_hat} - Label: {y}\n"
        file.write(content)
        file.write('*'*10+'\n')
    
    print('Accuracy:', acc / len(iter))
    file.close()


if __name__ == '__main__':


    output_path = f"./data/{dataset}/l2xs/l2x_k{k}"
    item_path = f'./data/{dataset}/l2xs/l2x_k{k}.pickle'
    model_path = f'./model/{dataset}/l2xs/l2x_k{k}.hdf5'

    # Run on blackbox predictions
    config.data_path = config.output_path[1]
    
    dg = DataGenerator(config)
    print(dg.data_path)
    dg.model_name = 'L2X'
    
    if dataset == 'hatex':
        dg.val_text = dg.val_text[:4000]
        dg.val_label = dg.val_label[:4000]

    dg.generate_data()
    print(dg.train_y.shape)
    

    preds, scores = L2X(True, dg, model_path, batch_size)
    write_pickle((preds, scores), item_path)

    file = open(output_path, 'w+')
    validation(scores, preds, file, k)


    