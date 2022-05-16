import data
import lstm
from numpy import squeeze

model, callbacks_list = lstm.create_model()
X,y = data.generate_data()


n_chars = len(data.generate_alphabet())
X = squeeze(X)
y = squeeze(y)
print(X.shape, y.shape)
n_in_seq_length = X.shape[1]
n_out_seq_length = y.shape[1]
# define LSTM configuration
n_batch = 100
n_epoch = 300
model.fit(X, y, epochs=n_epoch, batch_size=n_batch,callbacks=callbacks_list)
model.save('Final_model.hdf5')