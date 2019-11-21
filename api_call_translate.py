#Packages
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
import numpy as np
import pandas as pd

import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed
from attention_keras_thushv import AttentionLayer
from tensorflow.python.keras.models import Model

import tensorflow.keras as keras
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import os, sys
import pickle
import json
import time

#Reading data for tokenizer
spa = pd.read_csv('tokenizer_corpus.txt', header=None, sep='\n')
spa.columns = ['Content']
spa['English'] = spa['Content'].apply(lambda x: x.split('\t')[0])
spa['Spanish'] = spa['Content'].apply(lambda x: x.split('\t')[1])
spa = spa[['English','Spanish']]

#Storing as lists
english_text  = spa['English'].tolist()  
spanish_text  = spa['Spanish'].tolist()

#Adding start and end tokens in Spanish
spanish_text = ['sos ' + sent[:-1] + 'eos .'  if sent.endswith('.') else 'sos ' + sent + ' eos .' for sent in spanish_text]

""" Defining tokenizers """
en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
en_tokenizer.fit_on_texts(english_text)

sp_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
sp_tokenizer.fit_on_texts(spanish_text)

#Defining Vocab sizes
en_vsize = 9513   #max(en_tokenizer.index_word.keys()) + 1
sp_vsize = 22259  #max(sp_tokenizer.index_word.keys()) + 1

#Defining TimeSteps
en_timesteps, sp_timesteps = 20, 20

#defining functions
#Function 1
def sents2sequences(tokenizer, sentences, reverse=False, pad_length=None, padding_type='post'):

    encoded_text = tokenizer.texts_to_sequences(sentences)
    preproc_text = pad_sequences(encoded_text, padding=padding_type, maxlen=pad_length)

    if reverse:
        preproc_text = np.flip(preproc_text, axis=1)

    return preproc_text
	
# Function 2	
def define_nmt(hidden_size, batch_size, en_timesteps, en_vsize, sp_timesteps, sp_vsize):

    """ Defining a NMT model """

    # Define an input sequence and process it.

    if batch_size:
        encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(batch_shape=(batch_size, sp_timesteps - 1, sp_vsize), name='decoder_inputs')

    else:
        encoder_inputs = Input(shape=(en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(shape=(sp_timesteps - 1, sp_vsize), name='decoder_inputs')



    # Encoder GRU

    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru')
    encoder_out, encoder_state = encoder_gru(encoder_inputs)


    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru')
    decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)



    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])



    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])


    # Dense layer

    dense = Dense(sp_vsize, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)


    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    full_model.summary()



    """ Inference model """

    batch_size = 1



    """ Encoder (Inference) model """

    encoder_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inf_inputs')
    encoder_inf_out, encoder_inf_state = encoder_gru(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])



    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1, sp_vsize), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, hidden_size), name='encoder_inf_states')
    decoder_init_state = Input(batch_shape=(batch_size, hidden_size), name='decoder_init')



    decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)

    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])



    return full_model, encoder_model, decoder_model

# Function 3
def preprocess_data(en_tokenizer, sp_tokenizer, en_text, sp_text, en_timesteps, sp_timesteps):

    """ Preprocessing data and getting a sequence of word indices """

    en_seq = sents2sequences(en_tokenizer, en_text, reverse=False, padding_type='pre', pad_length=en_timesteps)
    sp_seq = sents2sequences(sp_tokenizer, sp_text, pad_length=sp_timesteps)
    
    return en_seq, sp_seq
	
# Function 5
def infer_nmt(encoder_model, decoder_model, test_en_seq, en_vsize, sp_vsize):

    """
    Infer logic

    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_en_seq: sequence of word ids
    :param en_vsize: int
    :param sp_vsize: int
    :return:

    """

    test_sp_seq = sents2sequences(sp_tokenizer, ['sos'], sp_vsize)
    
    test_en_onehot_seq = to_categorical(test_en_seq, num_classes=en_vsize)
    test_sp_onehot_seq = np.expand_dims(to_categorical(test_sp_seq, num_classes=sp_vsize), 1)



    enc_outs, enc_last_state = encoder_model.predict(test_en_onehot_seq)
    dec_state = enc_last_state

    attention_weights = []

    sp_text = ''

    for i in range(100):

        dec_out, attention, dec_state = decoder_model.predict([enc_outs, dec_state, test_sp_onehot_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]

        if dec_ind == 0:
            break

        test_sp_seq = sents2sequences(sp_tokenizer, [sp_tokenizer.index_word[dec_ind]], sp_vsize)
        
        test_sp_onehot_seq = np.expand_dims(to_categorical(test_sp_seq, num_classes=sp_vsize), 1)
        attention_weights.append((dec_ind, attention))

        sp_text += sp_tokenizer.index_word[dec_ind] + ' '

    return sp_text, attention_weights

hidden_size = 96
batch_size = 128
	
""" Defining the full model """
full_model, infer_enc_model, infer_dec_model = define_nmt(hidden_size=hidden_size, batch_size=batch_size,
        en_timesteps=en_timesteps, sp_timesteps=sp_timesteps,
        en_vsize=en_vsize, sp_vsize=sp_vsize)
		

# Loading weights
infer_dec_model.load_weights('decoder_weights_n.h5')
infer_enc_model.load_weights('encoder_weights_n.h5')

""" Inferring with trained model """
test_en = input('Enter English Sentence: ')
test_en_seq = sents2sequences(en_tokenizer, [test_en], pad_length=en_timesteps)

test_sp, attn_weights = infer_nmt(
	encoder_model=infer_enc_model, decoder_model=infer_dec_model,
	test_en_seq=test_en_seq, en_vsize=en_vsize, sp_vsize=sp_vsize)
#print(attn_weights)

#print('English: ', test_en)
print('Spanish: ', test_sp)
print('-'*50)










