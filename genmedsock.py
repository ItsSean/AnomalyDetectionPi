import io
import socket
import struct
import numpy as np
#import csv
#import shutil
from PIL import Image
from glob import glob
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
            #=np.concatenate(tf.nn.top_k((arr[:, i], arr[:, i].get_shape())[0]//2+1).values, median], axis=0)
            self.meds[i] = np.percentile(self.feats[:, i], 50.0, interpolation='midpoint')

    def updateModel(self, feat):
        self.feats = np.concatenate([self.feats, feat], axis=0)


server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)
md = medar()
connection = server_socket.accept()[0].makefile('rb')
try:
    bmo = InceptionV3(weights='imagenet', include_top=True)
    model = Model(inputs=bmo.input, outputs=bmo.layers[-2].output)
    while True:
        print("hm")
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        print("ok")
        stream = io.BytesIO()
        stream.write(connection.read(image_len))
        stream.seek(0)
        img = Image.open(stream)
        img = img.resize((299, 299))
        newFeat = image.img_to_array(img)
        newFeat = np.expand_dims(newFeat, axis = 0)
        newFeat = preprocess_input(newFeat)
        feat = model.predict(newFeat)
        md.updateModel(feat)
        print(feat)
finally:
    md.updateMedian()
    np.savetxt('medspi.csv', md.meds, delimiter=",")
    connection.close()
    server_socket.close()