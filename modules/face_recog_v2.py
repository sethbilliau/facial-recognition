"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.
When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.
Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.
* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.
Usage:
1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.
2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.
3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.
NOTE: This example requires scikit-learn to be installed! You can install it with pip:
$ pip3 install scikit-learn
"""

from FaceNet.architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw

######pathsandvairables#########
FACE_DATA = 'data/train/'
REQUIRED_SHAPE = (160,160)
FACE_ENCODER = InceptionResNetV2()
path = "facenet_keras_weights.h5"
FACE_ENCODER.load_weights(path)
face_detector = mtcnn.MTCNN()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
###############################

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_encodings_from_face_path(face_path = None, required_shape = REQUIRED_SHAPE,
                                face_encoder = FACE_ENCODER):
    
    # Import image 
    img_BGR = cv2.imread(face_path)
    if img_BGR is None:
        print('Wrong path:', face_path)
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    
    # Normalize images
    face = normalize(img_RGB)
    face = cv2.resize(face, required_shape)
    face_d = np.expand_dims(face, axis=0)
    
    # Get encoding
    encode = face_encoder.predict(face_d)[0]
    return encode


def train(face_data = FACE_DATA, n_neighbors = 3, knn_algo = 'ball_tree', model_save_path=None):

    person_list = []
    encodings = []
    
    # Iterate through all people
    for face_names in os.listdir(face_data):
        
        
        if face_names == '.DS_Store':
            continue
        person_dir = os.path.join(face_data,face_names)

        for image_name in os.listdir(person_dir):
            person_list.append(face_names)
            # Load images
            image_path = os.path.join(person_dir,image_name)

            # Get encodings from image path
            encode = get_encodings_from_face_path(image_path)
            encodings.append(encode)

    y = np.array(person_list)
    X = np.array(encodings)


    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    # Save the trained KNN classifier
    if model_save_path is not None:
        model_save_path = 'models/trained_knn_model_v2.clf'
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
            
    return knn_clf


def get_encodings_from_face_array(face, required_shape = REQUIRED_SHAPE,
                                face_encoder = FACE_ENCODER):

    # Normalize images
    face = normalize(face)
    face = cv2.resize(face, required_shape)
    face_d = np.expand_dims(face, axis=0)
    
    # Get encoding
    encode = face_encoder.predict(face_d)[0]
    return encode
