#Packages
#import collections
import os, sys
#import helper
import numpy as np
import re
#import project_tests as tests
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import sparse_categorical_crossentropy

#reading data
spa = pd.read_csv('spa.txt', sep='\n', header = None)

#Preprocessing
spa.columns=['Content']
text = spa['Content'].apply(lambda x: x[: x.find('CC-BY 2.0')])
text = text.str.strip()
spa['English'] = text.apply(lambda x: x.split('\t')[0])
spa['Spanish'] = text.apply(lambda x: x.split('\t')[1])
spa = spa[['English','Spanish']]
spa['English'] = spa['English'].str.lower()
spa['Spanish'] = spa['Spanish'].str.lower()

#Removing dots from starting and ending and numbers
spa['English'] = spa['English'].apply(lambda x: re.sub('[<>;+:!¡/\|?¿,.0-9@#$%^&*"]+' , '' , x))
spa['Spanish'] = spa['Spanish'].apply(lambda x: re.sub('[<>;+:!¡/\|?¿,.0-9@#$%^&*"]+' , '' , x))

#replacing hypen with a space
spa['English'] = spa['English'].apply(lambda x: re.sub('[-]+' , ' ' , x))
spa['Spanish'] = spa['Spanish'].apply(lambda x: re.sub('[-]+' , ' ' , x))
del(text)

#Shuffling dataset
spa = spa.sample(frac = 1)

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer
	
def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    return pad_sequences(x, maxlen=length, padding='post')
	
def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x, 50)
    preprocess_y = pad(preprocess_y, 50)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_spanish_sentences, english_tokenizer, spanish_tokenizer = preprocess(spa['English'].tolist(), spa['Spanish'].tolist())

max_english_sequence_length = preproc_english_sentences.shape[1]  #48
max_spanish_sequence_length = preproc_spanish_sentences.shape[1]  #53
english_vocab_size = len(english_tokenizer.word_index)+1   #13411
spanish_vocab_size = len(spanish_tokenizer.word_index)+1   #26266

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
	
def bd_model(input_shape, output_sequence_length, english_vocab_size, spanish_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement

    # Hyperparameters
    #learning_rate = 0.003
    
    # TODO: Build the layers
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape= input_shape[1:]))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(spanish_vocab_size, activation='softmax'))) 

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model

# TODO: Reshape the input
tmp_x = pad(preproc_english_sentences, 50)
tmp_x = tmp_x.reshape((-1, 50, 1))

# TODO: Train and Print prediction(s)
bd_rnn_model = bd_model(
    tmp_x.shape,
    50,
    len(english_tokenizer.word_index)+1,
    len(spanish_tokenizer.word_index)+1)
	
print(bd_rnn_model.summary())

bd_rnn_model.fit(tmp_x, preproc_spanish_sentences, batch_size=1024, epochs=5, validation_split=0.2)

#Saving model
bd_rnn_model.save('Bd_RNN_Model.h5')




