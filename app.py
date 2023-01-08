import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.preprocessing import image

category={
    0: 'Bean', 1: 'Bitter Gourd', 2: 'Bottle Gourd', 3 : 'Green Brinjal', 4: "Broccoli", 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower',
    9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13 : "Radish", 14: "Tomato"
}

test_dataset = [
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=051e821168b524b0f9f054a7e21778781&authkey=AUoxRlM91Q-KijUKJlqjcek&e=8ptqRE',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=0cb41cd00cc62489fb6ed772a86bc4ef2&authkey=AY_MYYoZ_-wJY-mO8qp-mRU&e=a0mrJX',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=04f7d4857f5c04c1ba2b8eed4dd66c048&authkey=AQplgrKv6vmYmaV53MJxeDY&e=VpRV7m',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=0d78b44bb314a4afb83e0820300eec818&authkey=AYnBWiNHgS59n2rilo1WPwI&e=ekSUCh',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=0d015ae32a07d482aaa65e7355b1832d8&authkey=AcUhMAjSfa8GWvnU5MfwS5w&e=3irSJo',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=0f8df0aa970cd446daf094e0dae762aa5&authkey=Ae0Etw2pok6IrAanfbL1D3A&e=RE1Dhb',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=05a9d06800e354f859a2dd2f6d3ba5330&authkey=AcWwNoRBLjzklHedjW8SYPs&e=EMihQB',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=05168592fd6d048648ccae9862f03af14&authkey=AX206hNjwLp8P-Sg82Mhgb4&e=WLV8jO',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=0c3b9e4b860ed44f5a9099decdafead82&authkey=AXm4Z7YtBxik9qljIfnBiVk&e=mvZ5yr',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=03bdaa4b1d6944a438a43e4a3c5f8bf28&authkey=AUYQSyVVFAPdFAAttK4gt7o&e=YjLeHt',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=08bb4241aa474470a825546011c48d774&authkey=AaY_JARxD8bA6WeegiGw-U4&e=tCPp9A',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?guestaccesstoken=MCkMjiXe8vOPyyFS1qVowCQAWI0XxRJj%2FTaOn8yHsnw%3D&folderid=2_087a163b811d6466190a23054701e022e&rev=1&e=QPhAnA',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=0b3e3c5e20aab4bb4895d5d3e6c449d22&authkey=ARddIe32w5mg-hkvGFHaqWo&e=8w8q9w',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=059936ef455a14426a445256adef2be79&authkey=AcND8Ivkh3WrcoBuV81L6Cc&e=uKqD1H',
    'https://binusianorg-my.sharepoint.com/personal/vito_minardi_binus_ac_id/_layouts/15/guestaccess.aspx?folderid=04da96e9b9fab4321b0f5298fe56cb81f&authkey=ARtPJKN58bB9ZePu48zpGhI&e=UePxcl'
]

thinking_word = [
    'I think this is it!',
    'Is this correct?',
    'Maybe this one',
    'Here is what I thought'
]

def predict_image(filename,model):
    st.write('----')
    st.subheader('🔎Prediction result')
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.

    with st.spinner('Which vegetable is this hmm..'):
        prediction = model.predict(img_processed)
        st.success(thinking_word[random.randint(0, len(thinking_word)-1)])
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

def main():
    st.image('./images/vege-background.jpg')
    st.title("Predict your vegetables!")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.write('This site predict these vegetables below, you can also click them to try the images from our dataset or use your own picture!')

    col1, col2, col3, col4, col5 = st.columns(5)

    for idx,name in enumerate(category.values()):
        if idx < len(category)*(1/5):
            with col1:
                st.write(f'[{name}]({test_dataset[idx]})')
        elif idx < len(category)*(2/5):
            with col2:
                st.write(f'[{name}]({test_dataset[idx]})')
        elif idx < len(category)*(3/5):
            with col3:
                st.write(f'[{name}]({test_dataset[idx]})')
        elif idx < len(category)*(4/5):
            with col4:
                st.write(f'[{name}]({test_dataset[idx]})')
        else:
            with col5:
                st.write(f'[{name}]({test_dataset[idx]})')

    # upload image
    st.write("----")
    st.subheader('📄Upload your picture')
    uploaded_file = st.file_uploader(label="Choose an image",accept_multiple_files=False,type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # load model
        model = tf.keras.models.load_model('models/mobilenet_model_v2.h5')
        predict_image(uploaded_file ,model)

if __name__ == '__main__':
    main()