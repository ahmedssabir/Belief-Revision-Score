import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from model import ResNet152
import tensorflow as tf

import os


image_dir = 'imgs'
import keras as K
from keras_applications.imagenet_utils import _obtain_input_shape

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def preprocess(x):
    x = resize(x, (224,224), mode='constant') * 255
    x = preprocess_input(x)
    if x.ndim == 3:
        x = np.expand_dims(x, 0)
    return x
model = ResNet152()

for img_file in os.listdir(image_dir):
    #img = mpimg.imread(image_dir + '/' + img_file)
    img = image.load_img(image_dir + '/' + img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print(img_file, decode_predictions(preds, top=1)[0])


