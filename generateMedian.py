import tensorflow as tf
import numpy as np
#import csv
#import shutil
from glob import glob
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.preprocessing import image
from keras.models import Model

class medar:
    def __init__(self):
        self.feats = np.zeros((0, 2048), dtype=np.float32)
        self.meds = np.zeros(2048, dtype=np.float32)

    def updateMedian(self):
        n = self.feats.shape
        for i in range(n[1]):
            print(i, self.feats[0, i])
            self.meds[i] = np.percentile(self.feats[:, i], 50.0, interpolation='midpoint')
    
    def updateModel(self, feat):
        self.feats = np.concatenate([self.feats, feat], axis=0)

#dogdir = 'doglabels.csv'

if __name__ == "__main__":
    #names = []
    #with open(dogdir) as cf:
        #readCSV = csv.reader(cf, delimiter=',')
        #for row in readCSV:
            #name = row[1]
            #names.append(name)
    md = medar()
    files = glob('datasets/ucsd/Train00?/*.tif')
    bmo = InceptionV3(weights='imagenet', include_top=True)
    model = Model(inputs=bmo.input, outputs=bmo.layers[-2].output)
    count = 0
    for infile in sorted(files):
        if count > 999:
            break
        img = image.load_img(infile, target_size=(299, 299))
        newFeat = image.img_to_array(img)
        newFeat = np.expand_dims(newFeat, axis = 0)
        newFeat = preprocess_input(newFeat)
        feat = model.predict(newFeat)
        md.updateModel(feat)
        print(count, feat)
        count += 1
    md.updateMedian()
    print(md.meds.shape)
    np.savetxt('meds1.csv', md.meds, delimiter=",")