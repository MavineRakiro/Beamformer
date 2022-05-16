from flask import Flask, current_app, request, jsonify
from predictor import Predictor
import logging
import data as datagram
import geohash
from numpy import squeeze
from keras.models import load_model
from gevent.pywsgi import WSGIServer


app = Flask(__name__)

model = load_model('models/weights-improvement-294-0.0418.hdf5')
print(model.summary())
predictor = Predictor(model=model, alphabet=datagram.generate_alphabet())

def converter(str_list):
    output = list()
    for string in str_list:
        str_parts = string.split(' ')
        output.append([float(str_parts[0]),float(str_parts[-1])])
    return output

def to_str(pred):
    prd = [str(p) for p in pred]
    strp = ' '.join(prd)
    return strp


@app.before_first_request
def setup_logging():
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.addHandler(logging.StreamHandler())
        app.logger.setLevel(logging.INFO)

@app.route('/', methods=['POST'])
def predict():
    try:
        input_coord_list = request.get_json()['coordinates']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400
    if not len(input_coord_list) == 5:
        return jsonify(status_code='400', msg='Check the coordinates sent'), 400

        # current_app.logger.info('Converted {}'.format(input_coord_list))
    predictions = predictor.predict(input_coord_list)
    encoded_predictions = predictor.invert(squeeze(predictions))
    decoded_predictions = geohash.decode(encoded_predictions)
    angle, distance = predictor.calulate_bearing(decoded_predictions)
    current_app.logger.info('Prediction: {0} {1} angle {2} distance {3}'.format(encoded_predictions, decoded_predictions,angle,distance))
    return jsonify(predictions=decoded_predictions,predstr = encoded_predictions, angle=angle, distance = distance)




if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), application=app, spawn=3)
    http_server.serve_forever()