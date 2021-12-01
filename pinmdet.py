import io
import socket
import struct
import pickle
import time
import math, csv
import glob, sys
import numpy as np
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import matplotlib.pyplot as plt

class Modar:
    def __init__(self):
        self.arr = np.zeros(2048, dtype=np.uint16) # each element i in arr corresponds to xis that are greater than medians.
        self.nar = np.zeros(2048, dtype=np.uint16) # each element i in nar corresponds to xis that are less than medians.
        self.arp = np.zeros(2048, dtype=np.float32)
        self.nrp = np.zeros(2048, dtype=np.float32) # arp and nrp to store odds
        self.feats = np.zeros((0, 2048), dtype=np.float32) # to which new features are concatenated.
    
    def binarizeFeat(self, feat, meds):
        # element wise, cond represents thresholded values by (val>median) ? 1 : 0
        return np.greater(feat, meds, dtype=np.float32)

    def updateArr(self, conds):
        # arr and nar store class counts
        self.arr = np.add(self.arr, conds.astype(np.uint16))
        self.nar = np.add(self.nar, np.logical_not(conds).astype(np.uint16))
    
    def logProbabilize(self):
        # turns the class counts into odds, and logarize
        self.arp = np.log1p(np.divide(self.arr, np.add(self.arr, self.nar)))
        self.nrp = np.log1p(np.divide(self.nar, np.add(self.arr, self.nar)))

def train(mdr, meds, feat):
    mdr.feats = np.concatenate([mdr.feats, feat], axis=0)
    mdr.updateArr(mdr.binarizeFeat(feat, meds))

mdr = Modar()
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)
connection = server_socket.accept()[0].makefile('rb')
try:
    bmo = InceptionV3(weights='imagenet', include_top=True) # loading the Inception model through Keras as a base
    model = Model(inputs=bmo.input, outputs=bmo.layers[-2].output) # extract the second last layer, containing linear unnormalized log probs, i.e 'logits'
    meddir = 'medspi.csv'
    meds = []
    with open(meddir) as cf:
        rf = csv.reader(cf, delimiter=',')
        for row in rf:
            med = row[0]
            meds.append(med)
    meds = np.asarray(meds, dtype=np.float32)
    count = 0
    start = time.time()
    while True:
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        stream = io.BytesIO()
        stream.write(connection.read(image_len))
        stream.seek(0)
        img = Image.open(stream)
        img = img.resize((299, 299))
        newFeat = image.img_to_array(img)
        newFeat = np.expand_dims(newFeat, axis = 0)
        newFeat = preprocess_input(newFeat)
        feat = model.predict(newFeat)
        train(mdr, meds, feat)
        if (time.time() - start) > 5:
            count += 1
            print(count, feat)
            img.save('norms/normal{}.jpg'.format(count))
            start = time.time()
    with open('store.pck1', 'wb') as f:
        mdr.logProbabilize()
        pickle.dump([mdr.arr, mdr.nar, mdr.arp, mdr.nrp, mdr.feats], f)

finally:
    connection.close()
    server_socket.close()