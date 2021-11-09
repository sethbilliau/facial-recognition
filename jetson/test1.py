from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np 
from modules import face_recog
import os


names = []
# Iterate over test images
for image_file in os.listdir("data/test/seth"):
    if image_file[len(image_file)-1] != 'g':
        continue
    full_file_path = os.path.join("data/test/seth", image_file)

    print("Looking for faces in {}".format(image_file))

    # Find all people in the image using a trained classifier model
    # Note: You can pass in either a classifier file name or a classifier model instance
    predictions = face_recog.predict(full_file_path, cropped_face = True, model_path="models/trained_knn_model.clf")

    # Print results on the console
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))

    # Display results overlaid on an image
    # face_recog.show_prediction_labels_on_image(os.path.join("data/test/claire", image_file), predictions)
    img_path = os.path.join("data/test/seth", image_file)
    
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    for name, (top, right, bottom, left) in predictions:
        names.append(name)
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        
    del draw

    plt.imshow(np.array(pil_image))
    plt.show()
    
