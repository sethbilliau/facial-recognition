from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib
import os
from modules import face_recog


def facial_recognition_from_folder(image_folder_path, dest_path, file_prefix, model_path = "models/trained_knn_model.clf"):

    counter = 1
    for image_file in os.listdir(image_folder_path):
        if image_file[len(image_file)-1] != 'g':
            continue
        full_file_path = os.path.join(image_folder_path, image_file)

        print("Looking for faces in {}".format(image_file))

        # Run predictions
        predictions = face_recog.predict(full_file_path, cropped_face = False, model_path=model_path)

        # Display results overlaid on an image
        img_path = os.path.join(image_folder_path, image_file)

        pil_image = Image.open(img_path).convert("RGB")
        figure, ax = plt.subplots(1)
        ax.imshow(np.array(pil_image))
        # draw = ImageDraw.Draw(pil_image)
        for name, (top, right, bottom, left) in predictions:
            # Draw a box around the face using matplotlib
            rect = matplotlib.patches.Rectangle((left,bottom),right - left,top - bottom, edgecolor='r', facecolor="none")
            ax.add_patch(rect)
            # There's a bug in Pillow where it blows up with non-UTF-8 text
            # when using the default bitmap font
            ax.text(left, bottom - 5, name,color='white', fontsize=10)

        plt.show()
        plt.savefig(dest_path + file_prefix + str(counter) + '.png')
        counter += 1
        
    return



facial_recognition_from_folder('data/images-custom/seth', 'data/images-custom/inf', 'sethdetected')