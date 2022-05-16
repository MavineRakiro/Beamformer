import requests, json, geohash

KERAS_REST_API_URL = "http://localhost:5000"
coord = [[-1.2339019775390625, 36.80351257324219], [-1.2352752685546875, 36.80351257324219], [-1.2339019775390625, 36.80213928222656], [-1.2339019775390625, 36.80076599121094], [-1.2339019775390625, 36.79939270019531]]

cord = [geohash.encode(val[0],val[1], 7) for val in coord]
print(cord)
cdd = [[-1.2339019775390625, 36.80351257324219], [-1.2352752685546875, 36.80351257324219], [-1.2339019775390625, 36.80213928222656], [-1.2339019775390625, 36.80076599121094], [-1.2339019775390625, 36.79939270019531]]

    # submit the request
def call_api(crd):
    payload = {"coordinates": crd}

    r = requests.post(KERAS_REST_API_URL, json=payload)

    print(r.text)
    # exit(1)
    return r
    # ensure the request was successful
i = 0
while i < 10:
    i += 1
    res = call_api(coord)
    print("Success",res.status_code==200)
    if res.status_code==200:
        response = json.loads(res.content)
        print(response['predictions'], response['angle'], response['predstr'])
        coord.append(response['predictions'])
        cord.append(response['predstr'])
        cdd.append(response['predictions'])
        # print(len(coord))
        coord = coord[-5:]
        cord = cord[-5:]
        # cord,coord = coord, cord

    # break

print(cdd)
print(cdd[int(len(cdd)/2)])
        # print(len(coord))

