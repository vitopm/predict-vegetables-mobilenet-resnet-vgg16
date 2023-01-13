import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.preprocessing import image
import os

@st.cache(allow_output_mutation=True)
def load_models():
    with st.spinner('🤖 : *reading deep learning model*'):
        models = [
            tf.keras.models.load_model('models/mobilenet_model_v2.h5'), #0
            tf.keras.models.load_model('models/resnet_model.h5'), #1
            tf.keras.models.load_model('models/vgg16_model.h5') #2
        ]
    return models

# load model
model = load_models()

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

vegetable_google =[
    'https://www.google.com/search?q=flat+beans&source=hp&ei=566_Y7BY4ZOaB77-tsgF&iflsig=AK50M_UAAAAAY7-892BtJROEFrDTLO_cYIwnIud59_P6&ved=0ahUKEwjw57HUucH8AhXhieYKHT6_DVkQ4dUDCAc&uact=5&oq=flat+beans&gs_lcp=Cgdnd3Mtd2l6EAMyBQguEIAEMgUIABCABDIGCAAQFhAeMgYIABAWEB4yBggAEBYQHjIGCAAQFhAeMgYIABAWEB4yBggAEBYQHjIGCAAQFhAeMgYIABAWEB46EQguEIMBEMcBELEDENEDEIAEOggILhCxAxCDAToLCC4QgAQQsQMQgwE6DgguEIMBENQCELEDEIAEOhEILhCABBCxAxCDARDHARDRAzoLCAAQgAQQsQMQgwE6CAgAEIAEELEDOgoIABCxAxCDARAKOggILhCABBCxAzoICAAQsQMQgwE6BwgAEIAEEAo6DQgAEIAEELEDEIMBEAo6CAgAEBYQHhAKUABYkQpg7QpoAHAAeACAAX-IAdUGkgEDOC4ymAEAoAEB&sclient=gws-wiz',
    'https://www.google.com/search?q=bitter+gourd&ei=6K6_Y7XAOoie3LUPhqOV0AI&gs_ssp=eJzj4tVP1zc0TCpPM841NkszYHRg8OJJyiwpSS1SSM8vLUoBAIzzCX8&oq=bitter&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQARgAMgoILhCDARCxAxBDMgsILhCABBDHARCvATIICAAQgAQQsQMyBAguEEMyCwguEIAEEMcBEK8BMgsILhCABBCxAxCDATIECC4QQzIICAAQgAQQsQMyCAgAEIAEELEDMgUIABCABDoKCAAQRxDWBBCwAzoHCAAQsAMQQzoNCAAQ5AIQ1gQQsAMYAToPCC4Q1AIQyAMQsAMQQxgCOgwILhDIAxCwAxBDGAI6CwgAEIAEELEDEIMBOg4ILhCxAxCDARDHARDRAzoMCC4Q6gIQtAIQQxgDOg8ILhDUAhDqAhC0AhBDGAM6DAgAEOoCELQCEEMYAzoHCC4Q1AIQQzoRCC4QgAQQsQMQgwEQxwEQ0QM6CggAELEDEIMBEEM6DgguEK8BEMcBEJIDEIAEOggIABCABBDJA0oECEEYAEoECEYYAVCnBlj9DWCJHWgEcAF4AIABb4gB8wSSAQM2LjGYAQCgAQGwAQfIARHAAQHaAQYIARABGAnaAQYIAhABGAjaAQQIAxgH&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=bottle+gourd&ei=EK-_Y-uTCZq24t4P__Oa4Aw&gs_ssp=eJzj4tTP1TcwzikwjDdg9OJJyi8pyUlVSM8vLUoBAF7kCAA&oq=bott&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQARgAMgQILhBDMgoIABCxAxCDARBDMhEILhCABBCxAxCDARDHARCvATIICAAQgAQQsQMyBQgAEIAEMgUIABCABDILCAAQgAQQsQMQgwEyCwgAEIAEELEDEIMBMgUIABCABDIFCAAQgAQ6CggAEEcQ1gQQsAM6BwguENQCEEM6BAgAEEM6CwguEIAEELEDEIMBOgoILhDHARDRAxBDOgoIABCxAxCDARAKOhEILhCABBCxAxCDARDHARDRAzoICC4QgAQQsQM6CwguELEDEMcBEK8BOgsILhCABBDHARCvAUoECEEYAEoECEYYAFDWBViMDmDAFWgEcAF4AIABZ4gBvQOSAQM0LjGYAQCgAQHIAQjAAQE&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=green+brinjal&ei=Oa-_Y5zvFMXK3LUPiNG6gAE&ved=0ahUKEwjc79L7ucH8AhVFJbcAHYioDhAQ4dUDCA4&uact=5&oq=green+brinjal&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIICAAQBxAeEBMyCAgAEAcQHhATMggIABAHEB4QEzIICAAQBxAeEBMyCAgAEAcQHhATMggIABAHEB4QEzIHCAAQgAQQEzIHCAAQgAQQEzIHCAAQgAQQEzIHCAAQgAQQEzoKCAAQRxDWBBCwAzoHCAAQsAMQQzoNCAAQ5AIQ1gQQsAMYAToMCC4QyAMQsAMQQxgCOg8ILhDUAhDIAxCwAxBDGAI6BggAEAcQHjoHCAAQgAQQDUoECEEYAEoECEYYAVCUBVjLCmD-C2gBcAF4AYABoQKIAbIFkgEFNS4wLjGYAQCgAQHIARDAAQHaAQYIARABGAnaAQYIAhABGAg&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=broccoli&ei=Pq-_Y7z3BMm-z7sPpdWk8A8&ved=0ahUKEwi8jvT9ucH8AhVJ33MBHaUqCf4Q4dUDCA4&uact=5&oq=broccoli&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIHCC4QsQMQQzIFCAAQgAQyBQgAEIAEMgUILhCABDIFCAAQgAQyBQguEIAEMgQIABBDMgQIABBDMgcIABCABBAKMgUIABCABDoKCAAQRxDWBBCwAzoHCAAQsAMQQzoMCC4QyAMQsAMQQxgBOgQILhBDOgsILhCABBCxAxCDAToHCC4Q1AIQQzoRCC4QgAQQsQMQgwEQxwEQ0QM6CAgAELEDEIMBOgsIABCABBCxAxCDAToICAAQgAQQsQM6BAgAEAM6CAguEIAEELEDOgoILhCABBCxAxAKOgoIABCABBCxAxAKSgQIQRgASgQIRhgBUPYGWJERYL0SaANwAXgBgAHDAYgBlgmSAQMyLjaYAQCgAQHIARTAAQHaAQYIARABGAg&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=cabbage&ei=UK-_Y_WIJ7r2z7sPrZGkyA8&ved=0ahUKEwj18OCGusH8AhU6-3MBHa0ICfkQ4dUDCA4&uact=5&oq=cabbage&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIHCC4QsQMQQzIECAAQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCC4QgAQyBQguEIAEMgUIABCABDIECAAQQzIECC4QQzoKCAAQRxDWBBCwAzoHCC4Q1AIQQzoICAAQgAQQsQM6EQguEIAEELEDEIMBEMcBENEDOgsIABCABBCxAxCDAToICC4QsQMQgwE6CgguEIAEELEDEAo6DQguENQCELEDEIAEEAo6EAguEIAEELEDEIMBENQCEAo6BwgAEIAEEAo6BQguELEDSgQIQRgASgQIRhgAULIFWNoJYLALaAJwAXgAgAFkiAHUBJIBAzYuMZgBAKABAcgBCMABAQ&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=capsicum&ei=W6-_Y9CtPO3sz7sP9I-skAE&gs_ssp=eJzj4tZP1zc0MkwvKKioNGB0YPDiSE4sKM5MLs0FAF4KB78&oq=capsi&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgcILhCxAxBDMg4ILhCDARDUAhCxAxCABDILCC4Q1AIQsQMQgAQyBwgAEIAEEAoyBQgAEIAEMgUIABCABDIFCC4QgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6CggAEEcQ1gQQsAM6BwgAELADEEM6DAguEMgDELADEEMYAToPCC4Q1AIQyAMQsAMQQxgBOgQILhBDOgQIABBDOhEILhCABBCxAxCDARDHARDRAzoICC4QsQMQgwE6CAgAEIAEELEDOgsIABCABBCxAxCDAToNCC4QgAQQsQMQgwEQCjoKCAAQgAQQsQMQCjoKCC4QxwEQ0QMQQzoNCC4QgwEQ1AIQsQMQQzoLCC4QgAQQsQMQ1AJKBAhBGABKBAhGGABQtgNYkgdghw5oAnABeACAAZEBiAGPBJIBAzMuMpgBAKABAcgBD8ABAdoBBAgBGAg&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=carrot&ei=Z6-_Y77jG-Wlmge635e4DA&ved=0ahUKEwj-stGRusH8AhXlkuYKHbrvBccQ4dUDCA4&uact=5&oq=carrot&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIECC4QQzILCC4QrwEQxwEQgAQyCAguEIAEELEDMgUIABCABDILCAAQgAQQsQMQgwEyBQguEIAEMgsILhCABBCxAxCDATIICAAQgAQQsQMyBQgAEIAEMgUIABCABDoKCAAQRxDWBBCwAzoHCAAQsAMQQzoNCAAQ5AIQ1gQQsAMYAToMCC4QyAMQsAMQQxgCOg8ILhDUAhDIAxCwAxBDGAI6CQgAEAUQHhDxBDoFCAAQogQ6BwguENQCEEM6BAgAEEM6EQguEIAEELEDEIMBEMcBENEDOggILhCxAxCDAToFCAAQsQM6CAgAELEDEIMBSgQIQRgASgQIRhgBUJMBWL4VYPwVaAFwAXgAgAGDAogB9AuSAQUyLjguMZgBAKABAcgBEcABAdoBBggBEAEYCdoBBggCEAEYCA&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=cauliflower&ei=eK-_Y4TLBuOE3LUP3--uUA&gs_ssp=eJzj4tTP1TcwNC03LzBgdGDw4k5OLM3JTMvJL08tAgBepAf2&oq=cauli&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgoILhCxAxCDARBDMgUIABCABDIFCAAQgAQyBAgAEEMyBQgAEIAEMgUIABCABDIECC4QQzIFCAAQgAQyBQgAEIAEMgUIABCABDoKCAAQRxDWBBCwAzoHCAAQsAMQQzoMCC4QyAMQsAMQQxgBOg8ILhDUAhDIAxCwAxBDGAE6BwguENQCEEM6EQguEIAEELEDEIMBEMcBENEDOgsIABCABBCxAxCDAToLCC4QgwEQsQMQgAQ6BQguEIAEOggIABCABBCxA0oECEEYAEoECEYYAVDWCFi3DmDvEmgDcAF4AIABswKIAcEHkgEHMS4yLjEuMZgBAKABAcgBFMABAdoBBggBEAEYCA&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=cucumber&ei=g6-_Y661LJ7Zz7sPn8-puAI&gs_ssp=eJzj4tTP1TcwNK0wKTJg9OJILk0uzU1KLQIAP60GbA&oq=cucu&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgoILhDUAhCxAxBDMgsIABCABBCxAxCDATILCC4QgwEQsQMQgAQyBAgAEEMyCAgAEIAEELEDMgQILhBDMg0IABCABBCxAxCDARAKMgsIABCABBCxAxCDATIFCAAQgAQyCAgAELEDEIMBOgoIABBHENYEELADOgcIABCwAxBDOg0IABDkAhDWBBCwAxgBOgwILhDIAxCwAxBDGAI6BwguENQCEEM6CwguEIAEELEDEIMBOgsIABCABBCxAxDJAzoICC4QgAQQsQM6BwgAEIAEEAo6BwguEIAEEApKBAhBGABKBAhGGAFQlwdYkg5g5hJoBHABeACAAV2IAZIDkgEBNZgBAKABAcgBEcABAdoBBggBEAEYCdoBBggCEAEYCA&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=papaya+fruit&ei=q6-_Y_i_GKvtz7sPhqCg-AI&gs_ssp=eJzj4tDP1TdIN0krMmD04ilILEisTFRIKyrNLAEAWT4Hzg&oq=papaya&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQARgBMhEILhCDARCvARDHARCxAxCABDIHCC4Q1AIQQzIKCAAQsQMQgwEQQzIECAAQQzIICAAQgAQQsQMyBQgAEIAEMgcILhDUAhBDMgUIABCABDIFCAAQgAQyBwgAEIAEEAo6CggAEEcQ1gQQsAM6BwgAELADEEM6DQgAEOQCENYEELADGAE6DAguEMgDELADEEMYAjoECC4QQzoLCAAQgAQQsQMQgwE6CAguELEDEIMBOhEILhCABBCxAxCDARDHARCvAToLCC4QgAQQsQMQgwE6EwguEIAEELEDEIMBEMcBEK8BEAo6DQgAEIAEELEDEIMBEApKBAhBGABKBAhGGAFQpAdYzRBg2BloAnABeACAAWuIAYoEkgEDNS4xmAEAoAEByAESwAEB2gEGCAEQARgJ2gEGCAIQARgI&sclient=gws-wiz-serp',
    'https://www.google.com/search?q=potato&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiWicm8usH8AhXSILcAHVsqD4AQ_AUoAXoECAEQAw&biw=1282&bih=989',
    'https://www.google.com/search?q=pumpkin&tbm=isch&ved=2ahUKEwizwMi-usH8AhXTV3wKHSiYBlAQ2-cCegQIABAA&oq=pumpkin&gs_lcp=CgNpbWcQAzIHCAAQsQMQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6CAgAEIAEELEDULcFWKAMYK0NaABwAHgAgAGmAYgBwQWSAQM2LjKYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=xa-_Y_PJKdOv8QOosJqABQ&bih=989&biw=1282',
    'https://www.google.com/search?q=radish&tbm=isch&chips=q:radish,g_1:vegetable:TyIwHucKFGk%3D&hl=en-US&sa=X&ved=2ahUKEwir5dvJusH8AhXaiNgFHaL_DjUQ4lYoAHoECAEQIw&biw=1282&bih=989',
    'https://www.google.com/search?q=tomato&tbm=isch&ved=2ahUKEwiuqp3MusH8AhUEWXwKHftkCucQ2-cCegQIABAA&oq=tomato&gs_lcp=CgNpbWcQAzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIICAAQgAQQsQMyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEOgcIABCxAxBDOgUIABCxA1DvBFjwB2DgCGgAcAB4AIABogGIAdIEkgEDNi4xmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=4q-_Y66xFISy8QP7yam4Dg&bih=989&biw=1282&hl=en-US'
]

thinking_robot = [
    'I think this is it!',
    'Is this correct?',
    'Maybe this one',
    'Here is what I thought',
    'This is what I think, is this correct?',
    'I\'d to be human than robot',
    'How does it feel like to be human?',
    'I don\'t really like vegetables, because I can\'t eat them',
    'Eat your veggie, kids!'
]

def top_prediction(prediction, top_num):
    with st.expander(f"Here are {top_num} most likely vegetables:"):
        top_confidence = np.flip(np.sort(prediction))[0]
        top_index = np.flip(np.argsort(prediction))[0]
        for i in range (0, top_num):
            st.write(f'{i+1}. {category[top_index[i]]} - {100*top_confidence[i]:.5f}%')

def predict_image(filename, model, user_input = True):
    if user_input is True:
        st.write('#### Your image')
        st.image(filename, caption=filename.name, width=300)
    else:
        st.write('#### Your image')
        st.image(filename, caption=filename, width=300)

    st.write('----')
    st.subheader('🔎Prediction result')
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.

    with st.spinner('🤖 : *Which vegetable is this hmm..*'):
        prediction_mobilenet = model[0].predict(img_processed)
        prediction_resnet = model[1].predict(img_processed)
        prediction_vgg16 = model[2].predict(img_processed)
        st.success(f'🤖 : *{thinking_robot[random.randint(0, len(thinking_robot)-1)]}*')
    index_mobilenet = np.argmax(prediction_mobilenet)
    index_resnet = np.argmax(prediction_resnet)
    index_vgg16 = np.argmax(prediction_vgg16)

    # show result
    col11, col12, col13 = st.columns(3)
    with col11:
        # mobilenet
        st.write(f'##### MobileNetV2 Prediction: {category[index_mobilenet]}')
        fig, ax = plt.subplots()
        plt.title(f"MobileNetV2 Prediction - {category[index_mobilenet]} - {100*np.max(prediction_mobilenet):.2f}%")
        plt.axis('off')
        plt.imshow(img_array)
        st.pyplot(fig)
        top_prediction(prediction_mobilenet,5)

    with col12: # resnet
        st.write(f'##### ResNet Prediction: {category[index_resnet]}')
        fig, ax = plt.subplots()
        plt.title(f"ResNet Prediction - {category[index_resnet]} - {100*np.max(prediction_resnet):.2f}%")
        plt.axis('off')
        plt.imshow(img_array)
        st.pyplot(fig)
        top_prediction(prediction_resnet,5)
    
    with col13: #vgg16
        st.write(f'##### VGG16 Prediction: {category[index_vgg16]}')
        fig, ax = plt.subplots()
        plt.title(f"VGG16 Prediction - {category[index_vgg16]} - {100*np.max(prediction_vgg16):.2f}%")
        plt.axis('off')
        plt.imshow(img_array)
        st.pyplot(fig)
        top_prediction(prediction_vgg16,5)

    st.write('----')
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆')

def main():
    st.image('./asset/images/vege-background.jpg')
    st.title("🤔What veggie is this?")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('*\"Broccoli or cauliflower?\"*')
    st.caption('*\"I have no idea.. but this website does!\"*')
    st.write('----')
    col1, col2 = st.columns([1,2])
    with col1:
        st.image('asset/images/butterfly meme (2).jpg', width=200)
    with col2:
        st.write('I know we are all tired of guessing or asking what kind of vegetable is it on top the shelf in your local supermarket. No worries, we are here to help!')
        st.write('Utilizing and comparing the best deep learning model trained with __MobileNetV2__, __ResNet__, and __VGG16__  algorithm, this website can predict vegetables from the list below. Click button below to see which vegetable names are available to predict.')

    with st.expander("💡Show vegetable list"):
        [col1, col2, col3, col4, col5] = st.columns(5)
        for idx,name in enumerate(category.values()):
            if idx < len(category)*(1/5):
                with col1:
                    st.write(f'- [{name}]({vegetable_google[idx]})')
            elif idx < len(category)*(2/5):
                with col2:
                    st.write(f'- [{name}]({vegetable_google[idx]})')
            elif idx < len(category)*(3/5):
                with col3:
                    st.write(f'- [{name}]({vegetable_google[idx]})')
            elif idx < len(category)*(4/5):
                with col4:
                    st.write(f'- [{name}]({vegetable_google[idx]})')
            else:
                with col5:
                    st.write(f'- [{name}]({vegetable_google[idx]})')

        st.caption('Dataset Source: [Vegetable Image Dataset - Kaggle](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)')

    # upload image
    st.write("----")
    st.subheader('🖼️Insert vegetable picture to predict')

    tab1, tab2, tab3 = st.tabs(["⬆️Upload photo", '📸Take photo', '🔎Select from dataset'])

    with tab1:
        uploaded_file = st.file_uploader(label="Choose an image",accept_multiple_files=False,type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            predict_image(uploaded_file ,model) #predict image

    with tab2:
        with st.expander('Click here to open camera'):
            uploaded_file = st.camera_input("Take a picture")
        if uploaded_file is not None:
            predict_image(uploaded_file ,model) #predict image
    with tab3:
        path = 'Vegetable Images'

        dataset_types = os.listdir(path)
        select_dataset_type = st.selectbox('Dataset type', [x for x in dataset_types])
        dataset_path = f'{path}/{select_dataset_type}'

        vegetables = os.listdir(dataset_path)
        select_vegetable = st.selectbox('Vegetable', [x for x in vegetables])
        vegetable_path = f'{dataset_path}/{select_vegetable}'

        vegetable_images = os.listdir(vegetable_path)
        select_vegetable_image = st.selectbox('Image name', [x for x in vegetable_images])
        vegetable_image_path = f'{vegetable_path}/{select_vegetable_image}'

        predict_image(vegetable_image_path, model, user_input= False)

if __name__ == '__main__':
    main()