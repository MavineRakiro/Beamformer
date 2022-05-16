from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.callbacks import ModelCheckpoint

# create LSTM
print('> Defining model... ')
def create_model():
    model = Sequential()
    model.add(LSTM(200, input_shape=(n_in_seq_length, n_chars)))
    model.add(RepeatVector(n_out_seq_length))
    model.add(LSTM(150, return_sequences=True, dropout=0.2))
    model.add(LSTM(300, return_sequences=True))
    # model.add(LSTM(300, return_sequences=True))
    model.add(TimeDistributed(Dense(n_chars, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    filepath1="accuracy-improvement-{epoch:02d}-{accuracy:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # checkpoint1 = ModelCheckpoint(filepath1, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    print(model.summary())
    return model, callbacks_list
    