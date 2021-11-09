# python 3.8

from wand.image import Image
import os
import glob

def convert_heic_to_jpg_from_directory(SourceFolder, TargetFolder, fileprefix):
    '''
    Convert all .HEIC files in a Source Folder to .jpg files. Store these new .jpg images in the 
    TargetFolder with fileprefix as the naming convention (ex. fileprefix1.jpg)
    
    :param SourceFolder: string with a path to the Source Folder that contains .HEIC files
    :param TargetFolder: string with a path to the Target Folder to save the .jpg files
    :param fileprefix: string with a prefix for naming new .jpg files
    :return: None
    '''
    print("Current working Directory:" + os.getcwd())
    
    # make target directory if needed
    if not os.path.isdir(TargetFolder): 
        os.mkdir(TargetFolder)
        
    # get list of all .HEIC files
    heicFilenamesList = glob.glob(SourceFolder + '/' + '*.HEIC')

    for counter, file in enumerate(heicFilenamesList):
        # Get target file and print
        TargetFile = TargetFolder + "/" + fileprefix + str(counter + 1) + ".jpg" 
        print(TargetFile)

        # Save targetfile
        img=Image(filename=file)
        img.format='jpg'
        img.save(filename=TargetFile)
        img.close()

    print('Done :)')
    return 

# TEST 
# convert_heic_to_jpg_from_directory('data/heic-test', 'data/heic-test', 'test') 


