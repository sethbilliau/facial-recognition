# python 3.8

# import the necessary packages
import face_recognition
import pickle
import cv2
import os
import numpy as np

from imutils import paths
from imutils import build_montages

from sklearn.cluster import DBSCAN

def get_face_encodings(dataset_path, encodings_pickle, detection_method):
    '''
    Get face encodings for all cropped face images in the dataset_path
    
    :param dataset_path: string with a path to a directory with cropped face images in .jpg format
    :param encodings_path: string with a path to pickle of encodings data/encodings.pickle)
    :param detection_method: string with the face detection model to use: either `hog` or `cnn`
    :return: None
    '''
    # add / to the end of dataset_path if necessary
    if dataset_path[len(dataset_path)-1] != '/': 
        dataset_path = dataset_path + '/'
    
        
    # grab the paths to the input images in our dataset, then initialize
    # out data list (which we'll soon populate)
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset_path))
    data = []


    # loop over the image paths
    for i, imagePath in enumerate(imagePaths):
        # load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        print(imagePath)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model=detection_method)
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # build a dictionary of the image path, bounding box location,
        # and facial encodings for the current image
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

    # dump the facial encodings data to disk
    print("[INFO] serializing encodings...")
    f = open(endocings_pickle, "wb")
    f.write(pickle.dumps(data))
    f.close()
    
    print('Done! :)')
    
    return 


# args = vars(ap.parse_args())
encodings_path = 'drive/MyDrive/data/encodings/encodings.pickle'
min_samples=5
jobs = -1


def DBSCAN_with_face_encodings(encodings_path, min_samples=5, jobs= -1):
    # load the serialized face encodings + bounding box locations from
    # disk, then extract the set of encodings to so we can cluster on
    # them
    print("[INFO] loading encodings...")
    data = pickle.loads(open(encodings_path, "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]

    # cluster the embeddings
    print("[INFO] clustering...")
    clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"], min_samples=min_samples)
    clt.fit(encodings)
    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    
    # loop over the unique face integers
    for labelID in labelIDs:
        # find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes
        # from the set
        print("[INFO] faces for face ID: {}".format(labelID))
        idxs = np.where(clt.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(25, len(idxs)),
            replace=False)
        # initialize the list of faces to include in the montage
        faces = []
        # loop over the sampled indexes
        for i in idxs:
            # load the input image and extract the face ROI
            image = cv2.imread(data[i]["imagePath"])
            (top, right, bottom, left) = data[i]["loc"]
            face = image[top:bottom, left:right]
            # force resize the face ROI to 96x96 and then add it to the
            # faces montage list
            face = cv2.resize(face, (96, 96))
            faces.append(face)

        # create a montage using 96x96 "tiles" with 5 rows and 5 columns
        montage = build_montages(faces, (96, 96), (5, 5))[0]

        # show the output montage
        title = "Face ID #{}".format(labelID)
        title = "Unknown Faces" if labelID == -1 else title
        print(title)
        cv2.imshow(montage)
        cv2.waitKey(0)
        


    encodings_df = pd.DataFrame(encodings)
    encodings_df
    
    return 
