from keras.models import load_model
from geopy.distance import vincenty
import numpy as np
import geohash
import data as dta
# import glob
import flask
from math import *
from pandas import read_csv

alpha = dta.generate_alphabet()

# initialize our Flask application and the Keras model

app = flask.Flask(__name__)
bts = read_csv('data/btx.csv')
# print(bts.values())
btss = list()
for val in bts.values:
    btss.append((float(val[0]), float(val[1])))


# print(btss)
def load_my_model():
    global model
    model = load_model('models/weights-improvement-294-0.0418.hdf5')
    print(model.summary())


# files = glob.glob(data.TEST_DATA_DIR+"/*.npy")


# load_my_model()

@app.route("/predict", methods=["POST"])
def predict():
    app.logger.debug("JSON received...")

    app.logger.debug(flask.request.json)

    data = {"success": False}

    if flask.request.json:

        mydata = flask.request.json  # will be
        files = ['test_data/382val0.npy']
        for file in files:
            print("-----------------------------File--------------------------------\n", file)
            print("-------------------------------------------------------------------\n")
            X, Y = dta.generate_test_data(file)
            X = np.squeeze(X)
            Y = np.squeeze(Y)
            X = X[np.newaxis, :, :]

            print("Making predictions")
            # print(X[0])
            input = X[0]
            input = input[np.newaxis, :, :]
            print(input.shape)
            vl = input[0][0]
            vl = np.array(vl)
            vl = vl[np.newaxis, :, :]
            print(vl.shape)
            pred = model.predict(x=vl, batch_size=1, verbose=2)
            print(pred)

        return flask.jsonify(mydata)

    else:
        return "no json received"


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_my_model()
    app.run(debug='on')

# files = ['test_data/382val0.npy']
# for file in files:
#     print("-----------------------------File--------------------------------\n", file)
#     print("-------------------------------------------------------------------\n")
#     X, Y = data.generate_test_data(file)
#     X = np.squeeze(X)
#     Y = np.squeeze(Y)
#     X = X[np.newaxis, :, :]
#
#     print("Making predictions")
#     # print(X[0])
#     input = X[0]
#     input = input[np.newaxis, :, :]
#     print(input.shape)
#     vl = input[0][0]
#     vl = np.array(vl)
#     vl = vl[np.newaxis, :, :]
#     print(vl.shape)
#     pred = model.predict(x=vl, batch_size=1, verbose=2)
#     print(pred)
#     for val in input:
#         print(val.shape)
#     # pred = [model.predict(x=val,batch_size=1,verbose=2) for val in input]
#     exit(1)
#     f1 = list()
#     per = 0
#     distances = list()
#     for k, v in enumerate(pred[0]):
#         splt = data.invert(v, alpha).split(' ')
#         first = geohash.decode(splt[0])
#         splt = data.invert(Y[k], alpha).split(' ')
#         fst = geohash.decode(splt[0])
#         f1.append(vincenty(first, fst).meters)
#         # distances.append([(k,vincenty(first, bt).meters) for bt in btss])
#     for v in f1:
#         if v == 0.0:
#             per += 1
#
#     mean = np.mean(f1)
#     variance, stddev = data.calculate_variance(f1, mean)
#     print("-------------------Prediction Summary----------------------")
#     print("Minimum Error (meters)    \t {0}".format(np.min(f1)))
#     print("Maximum Error (meters)    \t {0}".format(np.max(f1)))
#     print("Mean Error (meters)    \t\t {0}".format(mean))
#     print("Standard deviation     \t\t {0}".format(stddev))
#     print("Variance     \t\t\t {0}".format(variance))
#     print("Percentage correct    \t\t {0}".format((per / len(f1)) * 100))
#
#     # for key,val in enumerate(btss):
#     #
#     #     for k,v in enumerate(pred[0]):
#     #         splt = data.invert(v, alpha).split(' ')
#     #         first = geohash.decode(splt[0])
#     #         print()
#     #         print(key,k,vincenty(val,first).meters)
#     bts_used = btss[7]
#     pts = pred[0]
#     # [304:]
#     Aaltitude = 2000
#     Oppsite = 20000
#
#
#     def calcBearing(lat1, lon1, lat2, lon2):
#         dLon = lon2 - lon1
#         y = sin(dLon) * cos(lat2)
#         x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
#         return atan2(y, x)
#
#
#     for v in pts:
#         point = geohash.decode(data.invert(v, alpha).split(' ')[0])
#         # print(point)
#         # print(vincenty(bts_used,point).meters)
#
#         lon1, lat1, lon2, lat2 = map(radians, [bts_used[1], bts_used[0], point[1], point[0]])
#         dlon = lon1 - lon2
#         dlat = lat1 - lat2
#         a = sin(dlat / 2) ** 2 + cos(lat1 * cos(lat2)) * sin(dlon / 2) ** 2
#         c = 2 * atan2(sqrt(a), sqrt(1 - a))
#         Base = 6371 * c
#         Bearing = calcBearing(lat1, lon1, lat2, lon2)
#         Bearing = degrees(Bearing)
#
#         Base2 = Base * 1000
#         distance = Base * 2 + Oppsite * 2 / 2
#         Caltitude = Oppsite - Aaltitude
#
#         a = Oppsite / Base
#         b = atan(a)
#         c = degrees(b)
#
#         distance = distance / 1000
#         print("Bearing", Bearing)

# print(f1[304], f1[309])
# print(geohash.decode(data.invert(pred[0][304], alpha).split(' ')[0]), geohash.decode(data.invert(pred[0][309], alpha).split(' ')[0]))


# rs = np.array(distances)
# print(np.min(rs))






