import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.preprocessing import image
import os
import glob
import pandas as pd


def main():
    st.image('./asset/images/vege-background.jpg')
    st.title("🤔Exploratory Data Analysis")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('*\"What is inside this dataset?\"*')
    st.write('----')

    class_dirs = os.listdir("Vegetable Images/train") # list all directories inside "train" folder
    image_dict = {} # store image array(key) for every class(value)
    count_dict = {} # store count of files(key) for every class(value)

    for cls in class_dirs: # iterate over all class_dirs
        file_paths = glob.glob(f'Vegetable Images/train/{cls}/*') # get list of all paths inside the subdirectory
        count_dict[cls] = len(file_paths) # count number of files in each class and add it to count_dict
        image_path = random.choice(file_paths) # select random item from list of image paths
        image_dict[cls] = tf.keras.utils.load_img(image_path) # load image using keras utility function and save it in image_dict

    st.subheader("Data Distribution of training data across classes")
    df_count_train = pd.DataFrame({
    "class": count_dict.keys(), # keys of count_dict are class labels
    "count": count_dict.values(), # value of count_dict contain counts of each class
    })

    col11, col12 = st.columns([1,2])
    with col11:
        st.text(df_count_train)

    with col12:
        fig = plt.subplots()
        plt.figure(figsize=(15,12))
        df_count_train.plot.bar(x='class', y='count', title="Training Data Count per class")
        st.pyplot()
    
    st.write('----')
    st.subheader('Random sample from the dataset')
    # take random sample from each class
    fig = plt.figure(figsize=(15, 12))
    for i, (cls,img) in enumerate(image_dict.items()): # iterate over dictionary items (class label, image array)
        # create a subplot axis
        plt.subplot(5,3, i + 1)
        # plot each image
        # plt.tight_layout()
        plt.imshow(img)
        # set "class name" along with "image size" as title 
        plt.title(f'{cls}, {img.size}')
        plt.axis("off")
    st.pyplot(fig)

if __name__ == '__main__':
    main()