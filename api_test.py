# # from pymongo import MongoClient
# # import requests
# # import data
# # import json
# # import time
# # import numpy as np
# # from numpy import newaxis
# # from geopy.distance import vincenty
# # import geohash
# # from pandas import read_csv
# #
# #
# # KERAS_REST_API_URL = "http://localhost:5000"
# # bts = read_csv('data/btx.csv')
# # # print(bts.values())
# # btss = list()
# # for val in bts.values:
# #     btss.append((float(val[0]),float(val[1])))
# #
# # files = ['test_data/382val0.npy']
# # for file in files:
# #     print("-----------------------------File--------------------------------\n", file)
# #     print("-------------------------------------------------------------------\n")
# #     X, Y = data.generate_api_data(file)
# # i = 0
# # while i<1:
# #     for k,dst in enumerate(X):
# #         dt = [geohash.decode(val) for val in dst]
# #         y = geohash.decode(Y[k][0])
# #
# #         # print("lets see",Y[k][0])
# #         payload = {"coordinates": dt}
# #         # print(json.dumps(payload))
# #         # exit(1)
# #     # submit the request
# #         r = requests.post(KERAS_REST_API_URL, json=payload)
# #
# #     # ensure the request was successful
# #         print("Success",r.status_code==200)
# #         response = json.loads(r.content)
# #         # print(response['predictions'])
# #         print(k,dt, vincenty(y,response['predictions']).meters)
# #         time.sleep(1)
# #         # break
# #
# #     i+=1
# #     # break
# # # if r["success"]:
# # #     # loop over the predictions and display them
# # #     print(r)
# # #     # for (i, result) in enumerate(r["predictions"]):
# # #     #     print("{}. {}: {:.4f}".format(i + 1, result["label"],
# # #     #         result["probability"]))
# # #
# # # # otherwise, the request failed
# # # else:
# # #     print("Request failed")
# #
# #
# # # use 40 - 50 to test api
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pyroomacoustics as pra
#
# # Create a 4 by 6 metres shoe box room
# room = pra.ShoeBox([4,6])
#
# # Add a source somewhere in the room
# room.add_source([2.5, 4.5])
#
# # Create a linear array beamformer with 4 microphones
# # with angle 0 degrees and inter mic distance 10 cm
# R = pra.linear_2D_array([2, 1.5], 4, 0, 0.04)
# room.add_microphone_array(pra.Beamformer(R, room.fs))
#
# # Now compute the delay and sum weights for the beamformer
# room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])
#
# # plot the room and resulting beamformer
# room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
# plt.show()


# set alpha
# initialise w with random numbers
# set E to a large value (Emax)
# iter = 0
# Repeat for until E < Emax or iter < maxIter
#  E = 0
#  for all training patterns {(x, d)}
# output y = Wx
# w = w + alpha*(d-y)*x

import numpy as np
import matplotlib.pyplot as plt
import random

# weight_vector = np.zeros(3)
# w = np.array([0.1, 0.1, 0.1]) #25 weights

error_grp = []
iter_grp = []
# initialise the training patterns
training = [[np.array([-0.5, 1.2, -0.1]).transpose(), 0.2], [np.array([0.7, -0.5, -0.2]).transpose(), -0.8],
            [np.array([0.3, 1.2, 2.3]).transpose(), 0.8],
            [np.array([1.2, 0.8, 1.0]).transpose(), 0.4], [np.array([-0.5, 1.2, -0.1]).transpose(), -0.2],
            [np.array([1.0, -0.3, 0.5]).transpose(), -0.1]]


def lms(w):
    i = 0
    alpha = [0.1, 0.01, 0.001]
    Emax = 10000000000000000000000000
    maxIter = 500
    E = 0
    while ((i < maxIter) and (E < Emax)):
        E = 0
        for pair in training:
            y = np.dot(w.transpose(), pair[0])
            # print('y: ')
            # print(y)
            # print('==================')
            w = w + np.dot((alpha[2] * (pair[1] - y)), pair[0])
            # print('weight: ' + str(w))
            E = E + np.power((pair[1] - y), 2)
            # print('Error: ' + str(E))
        # Put the error in the array after going thru the whole training pattern
        error_grp.append(E)
        iter_grp.append(i)
        i = i + 1
    # print('i: ' + str(i))

    print('Final error: ' + str(E))
    print('final weight: ' + str(w))


w = []
for x in range(1):
    for y in range(3):
        val = round(random.random(), 1)
        print(val)
        w.append(val)
    weight_vector = np.array(w)
    print('Weight vector: ' + str(weight_vector))
    lms(weight_vector)
    w = []

# Draw the graph
plt.plot(iter_grp, error_grp)
plt.ylabel('Error')
plt.xlabel('No. of iterations')
plt.show()
# plt.savefig('0.01.png')