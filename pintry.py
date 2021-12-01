import io
import socket
import struct
import pickle
import math, csv
import glob, sys
import numpy as np
import time
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import matplotlib.pyplot as plt

class Modar:
    def __init__(self):
        with open('store.pck1', 'rb') as f:
            self.arr, self.nar, self.arp, self.nrp, self.feats = pickle.load(f) # arp and nrp contain likelihood  
    
    def binarizeFeat(self, feat, meds):
        # element wise, cond represents thresholded values by (val>median) ? 1 : 0
        return np.greater(feat, meds, dtype=np.float32)

    def updateArr(self, conds):
        # arr and nar store class counts
        self.arr = np.add(self.arr, conds.astype(np.uint16))
        self.nar = np.add(self.nar, np.logical_not(conds).astype(np.uint16))
        self.arp = np.log1p(np.divide(self.arr, np.add(self.arr, self.nar)))
        self.nrp = np.log1p(np.divide(self.nar, np.add(self.arr, self.nar)))

def graph(xs, logs, xz, weird):
    plt.plot(np.asarray(xs), np.asarray(logs), label="logprobs")
    plt.scatter(np.asarray(xz), np.asarray(weird), c='red', marker='X')
    plt.xlabel('iterations')
    plt.ylabel('vals')
    plt.title('Plot of log-prob')
    plt.legend()
    plt.show()

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
    logs = []
    xs = []
    weird = []
    xz = []
    flag = 0
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
        count += 1
        cond = mdr.binarizeFeat(feat, meds) # given an observed feature, binarize it
        mdr.updateArr(cond) # update the class counts
        # if the observed feature is binarized to be anomalous, take the log-likelihood from the anomalous class
        # , and vice versa
        logprob = np.where(cond, mdr.arp, mdr.nrp)
        slp = np.sum(logprob)
        print(count, slp)
        if count > 2 and slp < 1000 and (logs[-2] > logs[-1] and logs[-1] < slp):
            # weird
            img.save('anoms/anomalyat{}.jpg'.format(count-1))
            weird.append(logs[-1])
            xz.append(count-1)
        if (time.time() - start) > 5:
            img.save('pic{}.jpg'.format(count))
            start = time.time()
        prevlog = slp
        logs.append(slp)
        xs.append(count)
    graph(xs, logs, xz, weird)

finally:
    connection.close()
    server_socket.close()