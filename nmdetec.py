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
        self.nrp = np.zeros(2048, dtype=np.float32)
        self.feats = np.zeros((0, 2048), dtype=np.float32) # to which new features are concatenated.
    
    def binarizeFeat(self, feat, meds):
        # element wise, cond represents thresholded values by (val>median) ? 1 : 0
        return np.greater(feat, meds, dtype=np.float32)

    def updateArr(self, conds):
        self.arr = np.add(self.arr, conds.astype(np.uint16))
        self.nar = np.add(self.nar, np.logical_not(conds).astype(np.uint16))
    
    def logProbabilize(self):
        # turns the class counts into probs
        self.arp = np.divide(self.arr, np.add(self.arr, self.nar))
        self.nrp = np.divide(self.nar, np.add(self.arr, self.nar))

def imgGet(fp):
    img = image.load_img(fp, target_size=(299, 299))
    newFeat = image.img_to_array(img)
    newFeat = np.expand_dims(newFeat, axis = 0)
    newFeat = preprocess_input(newFeat)
    return newFeat

def train(mdr, meds, fps): # simply updates class counts
    count = 0
    for infile in sorted(fps):
        if count > 999:
            break
        newFeat = imgGet(infile)
        feat = model.predict(newFeat)
        mdr.feats = np.concatenate([mdr.feats, feat], axis=0)
        mdr.updateArr(mdr.binarizeFeat(feat, meds))
        count += 1
        print(count, feat)
    mdr.logProbabilize()

def test(mdr, meds, fps):
    count = 0
    logs = []
    xs = []
    for infile in sorted(fps):
        newFeat = imgGet(infile)
        feat = model.predict(newFeat) # extract logits
        cond = mdr.binarizeFeat(feat, meds) # binarize logits by median
        mdr.updateArr(cond)           # update class counts too
        logprob = np.where(cond, mdr.arp, mdr.nrp)
        slp = np.sum(logprob)
        logs.append(slp)
        xs.append(count)
        count += 1
        print(count, slp)
    return xs, logs

def graph(xs, logs):
    plt.plot(np.asarray(xs), np.asarray(logs), label="logprobs")
    plt.xlabel('iterations')
    plt.ylabel('vals')
    plt.title('Plot of logarized probs')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    files = glob.glob('datasets/ucsd/Train00?/*.tif') # using the UCSD Anomaly dataset
    bmo = InceptionV3(weights='imagenet', include_top=True) # loading the Inception model through Keras as a base
    model = Model(inputs=bmo.input, outputs=bmo.layers[-2].output) # extract the second last layer, containing linear unnormalized log probs, i.e 'logits'
    meddir = 'meds1.csv'
    meds = []
    with open(meddir) as cf:
        rf = csv.reader(cf, delimiter=',')
        for row in rf:
            med = row[0]
            meds.append(med)
    meds = np.asarray(meds, dtype=np.float32)
    mdr = Modar()
    train(mdr, meds, files)
    files = glob.glob('datasets/ucsd/Test0*/*.tif')
    xs, logs = test(mdr, meds, files)
    graph(xs, logs)