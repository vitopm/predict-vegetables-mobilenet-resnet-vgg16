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
    st.title("✍️ Data Preprocessing")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('*\"What are we going to do with this dataset?\"*')
    st.write('----')
    st.subheader('Data processing stage with resizing and rescaling operations')

    code = '''height, width = 224, 224

data_preprocess = keras.Sequential(
    name="data_preprocess",
    layers=[
        layers.Resizing(height, width), # shape preprocessing
        layers.Rescaling(1.0/255), # value preprocessing to normalize values
    ]
)

# Perform data processing on the train, val, test dataset
train_ds = train_data.map(lambda x, y: (data_preprocess(x), y))
val_ds = val_data.map(lambda x, y: (data_preprocess(x), y))
test_ds = test_data.map(lambda x, y: (data_preprocess(x), y))'''
    st.code(code, language='python')


if __name__ == '__main__':
    main()