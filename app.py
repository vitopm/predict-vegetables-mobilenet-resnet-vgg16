import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

# Tensorflow import
import tensorflow as tf
from tensorflow.keras.preprocessing import image

st.title("Predict your vegetables!")
st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')

category={
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3 : 'Brinjal', 4: "Broccoli", 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower',
    9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13 : "Radish", 14: "Tomato"
}

st.write('This site predict these vegetables below, you can also click them to try the images from our dataset or use your own picture!')
col1, col2, col3 = st.columns(3)
for idx,name in enumerate(category.values()):
    if idx < len(category)*(1/3):
        with col1:
            st.write(f'- {name}')
    elif idx < len(category)*(2/3):
        with col2:
            st.write(f'- {name}')
    else:
        with col3:
            st.write(f'- {name}')

# filename = './TEST/cabbage-1.jpg'
# filename = './TEST/cabbage-2.jpg'
# filename = './TEST/kacang.jpeg'
# filename = './TEST/tomato.jpg'
# filename = './TEST/timun.jpg'

def predict_image(filename,model):
    st.write('----')
    st.subheader('Prediction result')
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.   
    
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)

    # show result
    col11, col12 = st.columns(2)
    with col11:
        st.write('#### Your image')
        st.image(filename)

    with col12:
        st.write('#### Prediction')
        fig, ax = plt.subplots()
        plt.title("Prediction - {}".format(category[index]))
        plt.axis('off')
        plt.imshow(img_array)
        st.pyplot(fig)
    st.write('----')
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆')




# upload image
st.write("----")
st.subheader('Upload your picture')
uploaded_file = st.file_uploader(label="Choose an image",accept_multiple_files=False,type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # load model
    model = tf.keras.models.load_model('models/mobilenet_model_v2.h5')
    predict_image(uploaded_file ,model)
