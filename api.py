"""Flask application to serve up the model and make predictions"""

import flask
from keras.models import load_model
from numpy import newaxis
import numpy as np
import data as dta
import geohash


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def load_my_model():
    global model
    model = load_model('models/weights-improvement-294-0.0418.hdf5')
    # print(model.summary())

def prepare_data():
    pass

@app.route("/predict", methods=["POST"])

def predict():
    app.logger.debug("JSON received...")
    app.logger.debug(flask.request.json)
    data = {"success": False}
    if flask.request.json:
        mydata = flask.request.json  # will be
        arr = np.array(mydata.get("data"))
        arr = [geohash.encode(val[0], val[1],7) for val in arr]
        arr = dta.stringify([arr],[])
        arr,y = dta.one_hot_encode(arr[0][0],[],dta.generate_alphabet(),len(dta.generate_alphabet()))
        # print(np.array(arr).shape)
        arr = np.squeeze(arr)
        arr = arr[newaxis, :, :]
        print(arr.shape)
        print(arr)
        pred = model.predict(x=arr,batch_size=1,verbose=2)
        # files = ['test_data/382val0.npy']
        # for file in files:
        #     print("-----------------------------File--------------------------------\n", file)
        #     print("-------------------------------------------------------------------\n")
        #     X, Y = dta.generate_test_data(file)
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
            # pred = model.predict(x=vl, batch_size=1, verbose=2)
            # print(pred)
        # print(pred)

        # data["success"] = True
        return flask.jsonify(mydata)

    else:
        return "no json received"


def denormalise(val, w):
    return [((float(val[0])+1)*float(w[0])) , ((float(val[1])+1) * float(w[1]))]

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_my_model()
    app.run()
    
