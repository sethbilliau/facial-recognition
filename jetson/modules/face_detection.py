# python 3.8

# import libraries and dependencies
import os
import glob
import numpy as np 
import cv2
import mtcnn
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image

# draw each face separately
def get_faces(filename, faces_list, draw_faces = False):
    '''
    Given a filename for an image and a list of faces from an MTCNN detect_faces function call, 
    get all of the faces as cropped images in a list of numpy arrays. 
    
    :param filename: string with a path to an image in .jpg format
    :param faces_list: list of the results from calling detector.detect_faces() on the image

    :return faces: list of numpy arrays for cropped faces
    '''
    # load the image
    data = plt.imread(filename)
    
    # create an empty list to be filled with all faces
    faces = []
    for i in range(len(faces_list)):
        # get coordinates of face
        x1, y1, width, height = faces_list[i]['box']
        x2, y2 = x1 + width, y1 + height

        # plot face if draw_faces is True
        if draw_faces: 
            plt.subplot(1, len(faces_list), i+1)
            plt.imshow(data[y1:y2, x1:x2])
        
        # crop input image to face only and add to list 
        faces.append(data[y1:y2, x1:x2])
        
    # show the plot draw_faces is True
    if draw_faces: 
        plt.show()
        
    # return faces
    return faces

def face_detection_from_source_folder(SourceFolder, TargetFolder, fileprefix, size = 100):
    '''
    Convert all normal .jpg images in a SourceFolder to cropped face .jpg files. Store these new images in the 
    TargetFolder with fileprefix as the naming convention (ex. fileprefix1.jpg). 
    
    :param SourceFolder: string with a path to the Source Folder that contains normal images in .jpg format
    :param TargetFolder: string with a path to the Target Folder to save the cropped face .jpg files
    :param fileprefix: string with a prefix for naming new cropped face .jpg files
    :param SIZE: numeric for pixel size to resize faces - assumes 3:4 width:height ratio 
    :return: None
    '''
    # make target directory if needed
    if not os.path.isdir(TargetFolder): 
        os.mkdir(TargetFolder)

    # Get a list of all jpg image filenames in the source directory 
    jpgFilenamesList = glob.glob(SourceFolder + '/' + '*.jpg')
    
    # Create a face detector instance
    detector = MTCNN()

    # Iterate through all jpg image filenames
    filecounter = 1
    for filename in jpgFilenamesList:
        # Print the filename for the user's reference
        print(filename)

        # Read in the input image
        pixels = plt.imread(filename)

        # Detect faces in the image as dictionaries with MTCNN output
        faces = detector.detect_faces(pixels)

        # Get faces as cropped images from MTCNN output
        face_images = get_faces(filename, faces)

        # If there are no faces detected in the input image, continue to next image
        if len(face_images) == 0:
            continue

        # For each face in the 
        facecounter = 1
        for face in face_images: 
            # print shape of faces
            # print(face.shape)

            # If the face detection algorithm screws up, then skip this iteration 
            if face.shape[0] == 0 or face.shape[1] == 0 or face.shape[2] == 0:      
                continue

            # If the detected is smaller than 50 pixels, it's probably not what we want, so skip it. 
            if face.shape[0] < 50 or face.shape[1] < 50: 
                continue

            # Otherwise, process the image, resize it, and save it. 
            im = Image.fromarray(face)
            im = im.resize((3*size, 4*size))
            filename_save = TargetFolder + '/' + fileprefix + str(filecounter) +'face' + str(facecounter) + '.jpg'
            im.save(filename_save)

            # Iterate the facecounter 
            facecounter += 1

        # Iterate the filecounter 
        filecounter += 1
        
    print('Done! :)')

    return 

# USAGE
# face_detection_from_source_folder('data/images-custom/seth', 'data/images-custom/sethfaces', 'seth', size = 100)

    
    