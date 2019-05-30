import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras import layers
from keras.utils import to_categorical
from keras.models import Sequential
from keras import backend as K

#Data splitting:
DATASET_PATH = './news_en.csv'
dataset = pd.read_csv(DATASET_PATH)
training_set = dataset.loc[1:24036]
validating_set = dataset.loc[24037:32049]
testing_set = dataset.loc[32050:39599]

#word embeddings of input:
tokenizer_obj = Tokenizer()
all_data = np.concatenate([training_set['Body'].values, validating_set['Body'].values, testing_set['Body'].values])
tokenizer_obj.fit_on_texts(all_data)

max_length_x = sum([len(s.split()) for s in all_data])
max_length_x=int(max_length_x/39599)
vocab_size = len(tokenizer_obj.word_index) + 1

X_train_tokens = tokenizer_obj.texts_to_sequences(training_set['Body'])
X_validation_tokens = tokenizer_obj.texts_to_sequences(validating_set['Body'])
X_test_tokens = tokenizer_obj.texts_to_sequences(testing_set['Body'])

X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length_x, padding="pre")
X_validation_pad = pad_sequences(X_validation_tokens, maxlen=max_length_x, padding="pre")
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length_x, padding="pre")

#one_hot encoding of output:
#print(set(dataset['Kat']))
encoder = LabelEncoder()
training_set['Kat'] = training_set['Kat'].astype('category')
training_set['Kat'] = training_set['Kat'].cat.rename_categories([0, 1, 2, 3, 4])
validating_set['Kat'] = validating_set['Kat'].astype('category')
validating_set['Kat'] = validating_set['Kat'].cat.rename_categories([0, 1, 2, 3, 4])
testing_set['Kat'] = testing_set['Kat'].astype('category')
testing_set['Kat'] = testing_set['Kat'].cat.rename_categories([0, 1, 2, 3, 4])

Y_train = to_categorical(training_set['Kat'])
Y_validation = to_categorical(validating_set['Kat'])
Y_test = to_categorical(testing_set['Kat'])

#printing created maticies:
print(len(X_test_pad))
print(X_test_pad)
print(len(Y_test))
print(Y_test)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

EMBEDDING_SIZE = 100

#NEURAL NETWORK IMPLEMENTATION
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE, input_length=max_length_x))
model.add(layers.LSTM(units=64))
model.add(layers.Dense(80, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(55, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
model.summary()

model.fit(X_train_pad, Y_train, batch_size=112, epochs=5, validation_data=(X_validation_pad, Y_validation), verbose=1)
model.save('my_model.h5')
#model = load_model('my_model.h5')
loss, accuracy, f1_score, precision, recall = model.evaluate(X_validation_pad, Y_validation, verbose=0)
print(f1_score)
print(accuracy)

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test_pad, Y_test, verbose=0)
print(f1_score)
print(accuracy)