import streamlit as st

def main():
    st.image('./asset/images/vege-background.jpg')
    st.title("🧠CNN Architectures")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('*\"Mwahahaha i am the brain mwahahha\"*')

    tab1, tab2, tab3 = st.tabs(["MobileNetV2", "ResNet", "VGG16"])

    with tab1:
        st.subheader('Building MobileNetV2')
        st.write('with pretrained weight from imagenet')
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

        st.write('##### Compile the model with Adam optimizer, sparse categoritcal crossentropy loss function, with the metrics accuracy')
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

        st.write('##### MobileNet architecture')
        st.image('asset/images/mobilenet_model.png')

        with st.expander("View the complete base model"):
            st.image('asset/images/pretrained_mobilenet_model.png')

    with tab2:
        st.subheader('Building ResNet50V2')
        st.write('with pretrained weight from imagenet')
        code1 = '''
pretrained_resnet_model = tf.keras.applications.ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_tensor = (tf.keras.layers.Input(shape=(height,width,3)))
    )

pretrained_resnet_model.trainable=False

resnet_model = tf.keras.Sequential([
    pretrained_resnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(15, activation='softmax')
])
        '''
        st.code(code1, language='python')

        st.write('##### Compile the model with Adam optimizer, sparse categoritcal crossentropy loss function, with the metrics accuracy')
        code2 = '''
resnet_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
        '''
        st.code(code2, language='python')

        st.write('##### Summarize the model')
        code3 = '''
        mobilenet_model.summary()
        '''

        st.code(code3, language='python')


        st.text('''
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resnet50v2 (Functional)     (None, 7, 7, 2048)        23564800  
                                                                 
 global_average_pooling2d_2   (None, 2048)             0         
 (GlobalAveragePooling2D)                                        
                                                                 
 dense_4 (Dense)             (None, 15)                30735     
                                                                 
=================================================================
Total params: 23,595,535
Trainable params: 30,735
Non-trainable params: 23,564,800
_________________________________________________________________    
''')

        st.write('##### ResNet architecture')
        st.image('asset/images/resnet_model.png')

        with st.expander("View the complete base model"):
            st.image('asset/images/pretrained_resnet_model.png')


    with tab3:
        st.subheader('Building VGG16')
        st.write('with pretrained weight from imagenet')
        code1 = '''
pretrained_vgg16_model = tf.keras.applications.VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=[height,width, 3]
    )

pretrained_vgg16_model.trainable=False

vgg16_model = tf.keras.Sequential([
    pretrained_vgg16_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])
        '''
        st.code(code1, language='python')

        st.write('##### Compile the model with Adam optimizer, sparse categoritcal crossentropy loss function, with the metrics accuracy')
        code2 = '''
vgg16_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
        '''
        st.code(code2, language='python')

        st.write('##### Summarize the model')
        code3 = '''
        vgg16_model.summary()
        '''

        st.code(code3, language='python')


        st.text('''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688  
                                                                 
 global_average_pooling2d_1   (None, 512)              0         
 (GlobalAveragePooling2D)                                        
                                                                 
 dense_2 (Dense)             (None, 64)                32832     
                                                                 
 dense_3 (Dense)             (None, 15)                975       
                                                                 
=================================================================
Total params: 14,748,495
Trainable params: 33,807
Non-trainable params: 14,714,688
_________________________________________________________________    
''')

        st.write('##### VGG16 architecture')
        st.image('asset/images/vgg16_model.png')

        with st.expander("View the complete base model"):
            st.image('asset/images/pretrained_vgg16_model.png')

if __name__ == '__main__':
    main()