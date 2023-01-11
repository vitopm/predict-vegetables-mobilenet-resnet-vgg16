import streamlit as st

def main():
    st.image('./asset/images/vege-background.jpg')
    st.title("🔬Evaluation")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('*\"Have I been a good boy?\"*')

    tab1, tab2, tab3 = st.tabs(["MobileNetV2", "ResNet", "VGG16"])

    with tab1:
        st.write('### MobileNetV2')

        st.write('#### Traning Plot')
        st.image('asset/images/training_plot_mobilenet.png')
        st.info('Test Accuracy: 99.93%')
        st.write('----')
        
        st.write('#### Confusion Matrix')
        st.image('asset/images/confusion_matrix_mobilenet.png')
        
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
           4     0.9950    1.0000    0.9975       200
           5     1.0000    1.0000    1.0000       200
           6     1.0000    1.0000    1.0000       200
           7     1.0000    1.0000    1.0000       200
           8     1.0000    1.0000    1.0000       200
           9     1.0000    1.0000    1.0000       200
          10     1.0000    1.0000    1.0000       200
          11     1.0000    1.0000    1.0000       200
          12     1.0000    1.0000    1.0000       200
          13     1.0000    1.0000    1.0000       200
          14     1.0000    0.9950    0.9975       200

    accuracy                         0.9993      3000
   macro avg     0.9993    0.9993    0.9993      3000
weighted avg     0.9993    0.9993    0.9993      3000
        ''')
    with tab2:
        st.write('### ResNet50V2')

        st.write('#### Traning Plot')
        st.image('asset/images/training_plot_resnet.png')
        st.info('Test Accuracy: 99.73%')
        st.write('----')

        st.write('#### Confusion Matrix')
        st.image('asset/images/confusion_matrix_resnet.png')
        st.write('----')
        st.write('#### Classification Report')

        st.text('''
Classification Report:
----------------------
               precision    recall  f1-score   support

           0     0.9901    1.0000    0.9950       200
           1     1.0000    0.9900    0.9950       200
           2     1.0000    1.0000    1.0000       200
           3     0.9901    1.0000    0.9950       200
           4     0.9950    0.9900    0.9925       200
           5     1.0000    1.0000    1.0000       200
           6     1.0000    1.0000    1.0000       200
           7     1.0000    1.0000    1.0000       200
           8     1.0000    1.0000    1.0000       200
           9     0.9949    0.9850    0.9899       200
          10     0.9901    1.0000    0.9950       200
          11     1.0000    1.0000    1.0000       200
          12     1.0000    1.0000    1.0000       200
          13     1.0000    1.0000    1.0000       200
          14     1.0000    0.9950    0.9975       200

    accuracy                         0.9973      3000
   macro avg     0.9973    0.9973    0.9973      3000
weighted avg     0.9973    0.9973    0.9973      3000
        ''')

    with tab3:
        st.write('### VGG16')

        st.write('#### Traning Plot')
        st.image('asset/images/training_plot_vgg16.png')
        st.info('Test Accuracy: 99.47%')
        st.write('----')

        st.write('#### Confusion Matrix')
        st.image('asset/images/confusion_matrix_vgg16.png')
        st.write('----')
        st.write('#### Classification Report')

        st.text('''
Classification Report:
----------------------
               precision    recall  f1-score   support

           0     0.9950    0.9900    0.9925       200
           1     0.9901    1.0000    0.9950       200
           2     0.9950    1.0000    0.9975       200
           3     1.0000    1.0000    1.0000       200
           4     0.9900    0.9900    0.9900       200
           5     0.9950    0.9950    0.9950       200
           6     1.0000    0.9950    0.9975       200
           7     1.0000    1.0000    1.0000       200
           8     0.9950    0.9900    0.9925       200
           9     1.0000    0.9900    0.9950       200
          10     0.9803    0.9950    0.9876       200
          11     1.0000    0.9950    0.9975       200
          12     0.9900    0.9900    0.9900       200
          13     0.9950    1.0000    0.9975       200
          14     0.9950    0.9900    0.9925       200

    accuracy                         0.9947      3000
   macro avg     0.9947    0.9947    0.9947      3000
weighted avg     0.9947    0.9947    0.9947      3000
        ''')


    

    
if __name__ == '__main__':
    main()