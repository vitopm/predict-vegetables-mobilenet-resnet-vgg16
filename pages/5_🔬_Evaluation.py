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
    st.title("🔬Evaluation")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('*\"Broccoli or cauliflower?\"*')
    st.caption('*\"I have no idea.. but this website does!\"*')
    st.write('----')
    st.write('#### Confusion Matrix')
    st.image('asset\images\confusion_matrix.png')
    st.write('----')
    st.write('#### Classification Report')

    st.text('''
    Classification Report:
----------------------
               precision    recall  f1-score   support

           0     0.9950    1.0000    0.9975       200
           1     1.0000    0.9950    0.9975       200
           2     1.0000    1.0000    1.0000       200
           3     1.0000    1.0000    1.0000       200
           4     1.0000    1.0000    1.0000       200
           5     1.0000    1.0000    1.0000       200
           6     1.0000    1.0000    1.0000       200
           7     1.0000    1.0000    1.0000       200
           8     0.9950    1.0000    0.9975       200
           9     1.0000    1.0000    1.0000       200
          10     1.0000    1.0000    1.0000       200
          11     1.0000    1.0000    1.0000       200
          12     1.0000    0.9950    0.9975       200
          13     1.0000    1.0000    1.0000       200
          14     0.9950    0.9950    0.9950       200

    accuracy                         0.9990      3000
   macro avg     0.9990    0.9990    0.9990      3000
weighted avg     0.9990    0.9990    0.9990      3000
    ''')


    

    
if __name__ == '__main__':
    main()