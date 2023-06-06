from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import Embedding

# nltk.download('punkt')

def load_train_data(train_text_file):
    # Reading file
    file = open(train_text_file, 'r', encoding="utf8")
    training_text = file.read()
    training_text = training_text.replace('?', '.')
    training_text = training_text.replace('!', '.')
    training_text = training_text.replace('\n', ' ')
    training_text = training_text.lower()
    file.close()

    sequence_length = 5
    train_sequence = []
    # spliting file by "."
    splitted_sentences = training_text.split(".")

    for j in range(len(splitted_sentences)):
        base_text = splitted_sentences[j]
        if base_text.isspace():
            continue
        base_text.strip()
        # Getting only words, replacing uppercase to lowercase
        words = re.sub(r'\W+', ' ', base_text)
        words_lowercase = words.lower()
        tokens = word_tokenize(words_lowercase)

        if len(tokens) > sequence_length:
            # train sequence - based on how many words we predict next words
            for i in range(0, len(tokens) - sequence_length):
                words_sequence = tokens[i:sequence_length + i + 1]
                train_sequence.append(words_sequence)
        else:
            train_sequence.append(tokens)

    # replace words with tokens - numbers for model
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sequence)
    train_sequence = tokenizer.texts_to_sequences(train_sequence)

    train_sequence = pad_sequences(train_sequence, maxlen=sequence_length + 1, truncating='pre')

    # vocabulary size + 1 for the cause of padding
    number_of_available_words = len(tokenizer.word_counts) + 1

    train_sequence = np.array(train_sequence)
    train_X = train_sequence[:, :-1]
    train_y = train_sequence[:, -1]
    train_y = to_categorical(train_y, num_classes=number_of_available_words)

    return train_X, train_y, number_of_available_words, tokenizer


def build_and_compile_the_model(train_X, train_y, number_of_available_words):
    X_length = train_X.shape[1]
    model = Sequential()

    # building deep neural network
    model.add(Embedding(number_of_available_words, X_length, input_length=X_length))
    model.add(Bidirectional(LSTM(250, return_sequences=True)))
    model.add(LSTM(250))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(number_of_available_words, activation='softmax'))

    # compiling network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_y, epochs=10, verbose=1)

    return model


def predict_next_word(tokenizer, model, current_sequence, X_length):
    text_to_predict = current_sequence
    text_to_predict = text_to_predict.strip()
    text_to_predict = text_to_predict.lower()
    text_to_predict = tokenizer.texts_to_sequences([text_to_predict])[0]
    padded_text_to_predict = pad_sequences([text_to_predict], maxlen=X_length, truncating='pre')

    predicted_values = model.predict(padded_text_to_predict)[0]
    sorted_by_index_predicted_values = (model.predict(padded_text_to_predict)[0]).argsort()
    sorted_by_index_predicted_values = np.flip(sorted_by_index_predicted_values)[-5:]

    predicted_words = []
    for i in range(0, len(sorted_by_index_predicted_values)):
        index = sorted_by_index_predicted_values[i]
        # this is to ignore no such word
        if index == 0:
            continue
        word = tokenizer.index_word[index]
        predicted_words.append(word)

    return predicted_words
