# python 3.8

import pandas as pd
from shutil import copyfile
import os 
from sklearn.model_selection import train_test_split


def train_test_split_labeled_data(path_to_labeled_data, test_size):
    '''
    Loads in data from path, then performs train test split with sklearn
    
    :param path_to_labeled_data: string with a path to a labeled .csv file
    :param test_size: float between 0. and 1. for the proportion of total data to be put in the test set
    :return train, test: tuple of pandas data frames for train and test data sets
    '''
    # Load in labeled dataframe
    df = pd.read_csv(path_to_labeled_data).drop(columns = 'Unnamed: 0')

    # Drop none images
    df = df.drop(df[df.label == 'none'].index)
    df.reset_index(drop = True, inplace = True)

    train, test = train_test_split(df, test_size=test_size, random_state=42, shuffle=True, stratify = df.label)
    
    return train, test

def copy_images_to_new_folder(df, TargetFolder):
    '''
    Copy images from a labeled dataframe to a target folder. 
        
    :param df: pandas data frame with columns for filename and label (either train or test)
    :param TargetFolder: string with path to a target folder (either train or test)
    :return None: 
    '''
    # make target directory if needed
    if not os.path.isdir(TargetFolder): 
        os.mkdir(TargetFolder)

    for name in df.label.unique(): 
        # Filter dataframe by name 
        df_filtered_by_name = df[df.label == name]
        print(f'{name} dataframe rows: {df_filtered_by_name.shape[0]}')
        
        # Make a new folder for appropriate faces
        if not os.path.isdir(TargetFolder + '/' + name): 
            os.mkdir(TargetFolder + '/' + name)

        counter = 1
        for _, row in df_filtered_by_name.iterrows():
            copyfile(row['filename'], TargetFolder + '/'+ name + '/' + name + str(counter) + '.jpg')
            counter += 1
    return 



# test_size = 0.3
# path_to_labeled_data = 'drive/MyDrive/data/faces_labeled.csv'

# train, test = train_test_split_labeled_data(path_to_labeled_data, test_size)

# copy_images_to_new_folder(train, 'train')
# copy_images_to_new_folder(test, 'test')