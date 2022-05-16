import glob
import numpy as np
from numpy import argmax
import os

DATA_DIR = 'data'
TEST_DATA_DIR = 'test_data'

def invert(seq, alphabet):
	int_to_char = dict((i, c) for i, c in enumerate(alphabet))
	strings = list()
	for pattern in seq:
		string = int_to_char[argmax(pattern)]
		strings.append(string)
	return ''.join(strings)

def generate_alphabet():
    small_letters = map(chr, range(ord('a'), ord('z')+1))
    digits = map(chr, range(ord('0'), ord('9')+1))
    alpha = list(digits)
    alpha.extend(list(small_letters))
    alpha.append(" ")
    return alpha

def one_hot_encode(X, y, alphabet,max_int):
	Xenc = list()
	for seq in X:
		pattern = list()
		for index in seq[-4:]:
			vector = [0 for _ in range(max_int)]
			vector[alphabet.index(index)] = 1
			pattern.append(vector)
		Xenc.append(pattern)
	yenc = list()
	for seq in y:
		pattern = list()
		for index in seq[-4:]:
			vector = [0 for _ in range(max_int)]
			vector[alphabet.index(index)] = 1
			pattern.append(vector)
		yenc.append(pattern)
	return Xenc, yenc

def stringify(X,y):
    Xstr = list()
    for pattern in X:
        strp = ' '.join(pattern)
        Xstr.append(strp)
    # max_length = ceil(log10(n_numbers * (largest + 1)))
    ystr = list()
    for pattern in y:
        strp = ' '.join(pattern)

        ystr.append(strp)
    return Xstr, ystr

def read_file(file,dir):
    return np.load(os.path.join(dir, file))


def feature_label(data,feature_size, label_size):
    x = np.zeros((data.shape[0] - feature_size - label_size + 1, feature_size), dtype=object)
    y = np.zeros((data.shape[0] - feature_size - label_size + 1, label_size), dtype=object)
    # print(x.shape, y.shape)
    for i in range(len(data) - feature_size - label_size + 1):
        x[i] = data[i:i + feature_size]
        y[i] = data[i + feature_size :i + feature_size + label_size]

    # print("Check here", x.shape, y.shape)
    return x, y

# print(read_file('79val0.npy',DATA_DIR))

def find_differences(numbers,mean):
    diff = list()
    for num in numbers:
        diff.append(num - mean)
    return diff

def calculate_variance(numbers,mean):
    diff = find_differences(numbers,mean)
    # Find the squared differences
    squared_diff = []
    for d in diff:
        squared_diff.append(d**2)
    # Find the variance
    sum_squared_diff = sum(squared_diff)
    variance = sum_squared_diff / len(numbers)
    return variance,variance**0.5

def generate_data():
    alpha = generate_alphabet()
    files = glob.glob(DATA_DIR+"/*.npy")
    # print(files)
    Features = list()
    Labels = list()
    for file in files:
        data = read_file(file, '')
        xx, yy = feature_label(data, 5, 1)
        xx, yy = stringify(xx, yy)
        FT,LB = list(), list()
        for i in range(len(xx)):
            X,y = one_hot_encode(xx[i],yy[i],alpha,len(alpha))
            FT.append(X)
            LB.append(y)
        Features.extend(FT)
        Labels.extend(LB)
    return Features,Labels

# def generate_test_data():
#     alpha = generate_alphabet()
#     files = glob.glob(TEST_DATA_DIR+"/*.npy")
#     Features = list()
#     Labels = list()
#     for file in files:
#         data = read_file(file, '')
#         xx, yy = feature_label(data, 5, 1)
#         xx, yy = stringify(xx, yy)
#         FT,LB = list(), list()
#         for i in range(len(xx)):
#             X,y = one_hot_encode(xx[i],yy[i],alpha,len(alpha))
#             FT.append(X)
#             LB.append(y)
#         Features.extend(FT)
#         Labels.extend(LB)
#     return Features,Labels

def generate_test_data(file):
    alpha = generate_alphabet()
    # files = glob.glob(TEST_DATA_DIR+"/*.npy")
    Features = list()
    Labels = list()
    # for file in files:
    data = read_file(file, '')
    xx, yy = feature_label(data, 5, 1)
    xx, yy = stringify(xx, yy)
    FT,LB = list(), list()
    for i in range(len(xx)):
        X,y = one_hot_encode(xx[i],yy[i],alpha,len(alpha))
        FT.append(X)
        LB.append(y)
    Features.extend(FT)
    Labels.extend(LB)
    return Features,Labels

def generate_api_data(file):
    alpha = generate_alphabet()
    # files = glob.glob(TEST_DATA_DIR+"/*.npy")
    Features = list()
    Labels = list()
    # for file in files:
    data = read_file(file, '')
    xx, yy = feature_label(data, 5, 1)
    return xx,yy
    # xx, yy = stringify(xx, yy)
    # FT,LB = list(), list()
    # for i in range(len(xx)):
    #     X,y = one_hot_encode(xx[i],yy[i],alpha,len(alpha))
    #     FT.append(X)
    #     LB.append(y)
    # Features.extend(FT)
    # Labels.extend(LB)
    # return Features,Labels

generate_data()