from __future__ import division, absolute_import
import re
import numpy as np
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import cv2

#---------------------------------------------------adit deshpande
CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
SIZE_FACE = 48
EMOTIONS = ['angry', 'happy', 'sad', 'surprised', 'neutral']
SAVE_DIRECTORY = './temp/'
SAVE_MODEL_FILENAME = './temp/saved.tf'
SAVE_DATASET_IMAGES_FILENAME = './temp/data_training.npy'
SAVE_DATASET_LABELS_FILENAME = './temp/labels_training.npy'
SAVE_DATASET_IMAGES_TEST_FILENAME = './temp/data_testing.npy'
SAVE_DATASET_LABELS_TEST_FILENAME = './temp/labels_testing.npy'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def format_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    if not len(faces) > 0:
        #print "No hay caras"
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size


    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    # print image.shape
    print(image.shape)
    return image
training_data = np.load(SAVE_DATASET_IMAGES_FILENAME)
training_labels = np.load( SAVE_DATASET_LABELS_FILENAME)
testing_data = np.load(SAVE_DATASET_IMAGES_TEST_FILENAME)
testing_labels = np.load(SAVE_DATASET_LABELS_TEST_FILENAME)

training_data = training_data.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
testing_data= testing_data.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
training_labels = training_labels.reshape([-1, len(EMOTIONS)])
testing_labels = testing_labels.reshape([-1, len(EMOTIONS)])


network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1])
network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 3,strides=2)
network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 3,strides=2)
network = conv_2d(network, 128, 4, activation='relu')
network = dropout(network, 0.3)
network = fully_connected(network,3072,activation='relu')
network = fully_connected(network, len(EMOTIONS), activation='softmax')
network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)
model = tflearn.DNN(network, checkpoint_path= './temp/checkpoint.ckpt',
                         max_checkpoints=1, tensorboard_verbose=2)

if os.path.exists('{}.meta'.format(SAVE_MODEL_FILENAME)):
    model.load(SAVE_MODEL_FILENAME)
    print("model loaded")
else:
    print("model load failed!")

model.fit(
    training_data, training_labels,
            validation_set = (testing_data, testing_labels),
            n_epoch = 25,
            batch_size = 50,
            shuffle = True,
            show_metric = True,
            snapshot_step = 200,
            snapshot_epoch = True,
            run_id = 'emotion_recognition1'
        )

model.save(SAVE_MODEL_FILENAME)
print('[+] Model trained and saved at '+SAVE_MODEL_FILENAME)

# ----------------------------------------------------------------------------------------=
