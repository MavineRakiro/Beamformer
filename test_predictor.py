from predictor import Predictor
from keras.models import load_model
import data
import geohash
from numpy import squeeze

model = load_model('models/weights-improvement-294-0.0418.hdf5')
print(model.summary())

pred = Predictor(model=model,alphabet=data.generate_alphabet())
X,Y = data.generate_api_data('test_data/382val0.npy')
dt = [geohash.decode(val) for val in X[0]]
predicted = pred.predict(dt)
print(pred.invert(squeeze(predicted)))
print(predicted.shape)