"""predictor class that implement model.predict method"""
import numpy as np
import data as dt
import geohash
from pandas import read_csv
from math import *

def get_btss():
    bts = read_csv('data/btx.csv')
    # print(bts.values())
    btss = list()
    for val in bts.values:
        btss.append((float(val[0]), float(val[1])))
    return btss

class Predictor:
    def __init__(self, model: object, alphabet: object):
        self.model = model
        self.alphabet = alphabet
        self.btss = get_btss()

    def _get_feature_vector(self, input_coord_list):
        arr = np.array(input_coord_list)
        arr = [geohash.encode(val[0], val[1],7) for val in arr]
        arrx,arry = dt.stringify([arr],[])
        arr,y = dt.one_hot_encode(arrx[0],[],self.alphabet,len(self.alphabet))
        arr = np.squeeze(arr)
        featureset = arr[np.newaxis, :, :]
        return featureset

    def _get_feature_vector_encoded(self, arr):
        arrx,arry = dt.stringify([arr],[])
        arr,y = dt.one_hot_encode(arrx[0],[],self.alphabet,len(self.alphabet))
        arr = np.squeeze(arr)
        featureset = arr[np.newaxis, :, :]
        return featureset

    def _calcBearing(self, lat1,lon1,lat2,lon2):
        dLon = lon2-lon1
        y = sin(dLon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
        return atan2(y,x)

    def predict(self, input_coord_list):
        if isinstance(input_coord_list[0], str):
            featureset = self._get_feature_vector_encoded(input_coord_list)
        else:
            featureset = self._get_feature_vector(input_coord_list)

        assert featureset.shape[0] > 0

        result = self.model.predict(featureset)

        return result

    def calulate_bearing(self,point):
        bts_used = self.btss[7]
        # print("Point supplied",point)
        # bts_used = [-1.2407684326171875, 36.79252624511719]
        Aaltitude = 2000
        Oppsite = 20000
        # point = geohash.decode(predicted)
        lon1, lat1, lon2, lat2 = map(radians, [bts_used[1], bts_used[0], point[1], point[0]])
        dlon = lon1 - lon2
        dlat = lat1 - lat2
        a = sin(dlat / 2) ** 2 + cos(lat1 * cos(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        Base = 6371 * c
        Bearing = self._calcBearing(lat1, lon1, lat2, lon2)
        Bearing = degrees(Bearing)

        Base2 = Base * 1000
        distance = Base * 2 + Oppsite * 2 / 2
        Caltitude = Oppsite - Aaltitude

        a = Oppsite / Base
        b = atan(a)
        c = degrees(b)

        distance = distance / 1000
        return Bearing, distance


    def invert(self, predicted):
        return dt.invert(predicted, alphabet=self.alphabet)
