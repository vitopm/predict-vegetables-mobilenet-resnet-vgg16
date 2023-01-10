import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.preprocessing import image
import os
import glob
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, ReLU, Softmax, BatchNormalization, Dropout
from tensorflow.random import set_seed
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report


def main():
    st.image('./asset/images/vege-background.jpg')
    st.title("🧠MobileNetV2")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('*\"Broccoli or cauliflower?\"*')
    st.caption('*\"I have no idea.. but this website does!\"*')
    st.write('----')
    st.subheader('Building MobileNetV2')
    st.write('with pretrained with from imagenet')
    code1 = '''
    pretrained_mobilenet_model = tf.keras.applications.MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=[height,width, 3])

pretrained_mobilenet_model.trainable=False

mobilenet_model = tf.keras.Sequential([
        pretrained_mobilenet_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(15, activation='softmax')
    ])
    '''
    st.code(code1, language='python')

    st.write('##### Compile the model with Adam optimizer, spares categoritcal crossentropy loss function, with the metrics accuracy')
    code2 = '''
    mobilenet_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    '''
    st.code(code2, language='python')

    st.write('##### Summarize the model')
    code3 = '''
    mobilenet_model.summary()
    '''

    st.code(code3, language='python')

    height, width = 224, 224

    pretrained_mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=[height,width, 3])
    pretrained_mobilenet_model.trainable=False
    mobilenet_model = tf.keras.Sequential([
        pretrained_mobilenet_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(15, activation='softmax')
    ])

    st.text('''
    Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 mobilenetv2_1.00_224 (Funct  (None, 7, 7, 1280)       2257984   
 ional)                                                          
                                                                 
 global_average_pooling2d (G  (None, 1280)             0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 15)                19215     
                                                                 
=================================================================
Total params: 2,277,199
Trainable params: 19,215
Non-trainable params: 2,257,984
_________________________________________________________________
    ''')

    st.write('##### Draw the architecture')
    
    st.image('mobilenet_model.png')

    
if __name__ == '__main__':
    main()