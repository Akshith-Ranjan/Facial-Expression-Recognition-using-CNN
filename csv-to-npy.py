import numpy as np
import pandas as pd
import cv2

SIZE_FACE = 48
EMOTIONS = ['angry', 'happy', 'sad', 'surprised', 'neutral']
CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
READ_CSV_AT="./temp/train.csv"
SAVE_IMG_AT="./temp/data_training.npy"
SAVE_LAB_AT="./temp/labels_training.npy"
# --------------------------------------------------------------------------
anger=0
happy=0
sad=0
surprised=0
neutral=0
#------------------------------------------------

def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d

def data_to_image(data):
    #print data
    image = np.fromstring(str(data), dtype = np.uint8, sep = ' ').reshape((SIZE_FACE, SIZE_FACE))


    # data_image = Image.fromarray(data_image).convert('RGB')
    #
    # image = np.array(data_image)[:, :, ::-1].copy()



    gray_border = np.zeros((150, 150), np.uint8)
    gray_border[:,:] = 200

    gray_border[int((150 / 2) - (SIZE_FACE/2)):int((150/2)+(SIZE_FACE/2)), int((150/2)-(SIZE_FACE/2)):int((150/2)+(SIZE_FACE/2))] = image
    image = gray_border
    # faces = cascade_classifier.detectMultiScale(image, scaleFactor = 2,minNeighbors = 5)
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    # None is we don't found an image
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
    return image



data = pd.read_csv(READ_CSV_AT)

labels = []
images = []
index = 1
total = data.shape[0]
for index, row in data.iterrows():
    print('---')
    print(row['emotion'])
    if row['emotion'] != 2 and row['emotion'] !=1:

        print(row['emotion'])
        print('---')
        if row['emotion']== 3:
            row['emotion']=1
            happy +=1
        elif row['emotion']==4:
            row['emotion']=2
            sad += 1
        elif row['emotion']==5:
            row['emotion']=3
            surprised += 1
        elif row['emotion']==6:
            row['emotion']=4
            neutral += 1
        elif row['emotion']==0:
            anger+=1


        emotion = emotion_to_vec(row['emotion'])
        image = data_to_image(row['pixels'])
        if image is not None:
            # print(np.where(emotion==1.0))
            labels.append(emotion)
            images.append(image)
            #labels.append(emotion)
            #images.append(flip_image(image))
        else:
            # print "Error"
            print("Error")

        index += 1
        print ("Progreso: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))
print ("anger ={}".format(anger))
print ("happy ={}".format(happy))
print ("surprised ={}".format(surprised))
print ("sad ={}".format(sad))
print ("neutral ={}".format(neutral))
print ("Total: " + str(len(images)))
np.save(SAVE_IMG_AT, images)
np.save(SAVE_LAB_AT, labels)

