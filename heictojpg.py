# python 3.8

# This script converts images from heic to jpg format

from wand.image import Image
import os

print("Current working Directory:" + os.getcwd())

SourceFolder = input("Enter source directory:")
    
TargetFolder = input("Enter target directory:")
# make target directory if needed
if not os.path.isdir(TargetFolder): 
    os.mkdir(TargetFolder)

fileprefix = input("Enter file prefix:")

# convert 
counter = 1
for file in os.listdir(SourceFolder):
    if file == ".DS_Store": 
        continue
    SourceFile=SourceFolder + "/" + file
    TargetFile=TargetFolder + "/" + fileprefix + str(counter) + ".JPG" 
    print(TargetFile)

    counter += 1
    img=Image(filename=SourceFile)
    img.format='jpg'
    img.save(filename=TargetFile)
    img.close()





