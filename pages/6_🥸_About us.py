import streamlit as st

def main():
    st.image('./asset/images/vege-background.jpg')
    st.title("🥸 About us")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('Made by students, for students, with the help of internet')

    st.write('----')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('### 👨🏻‍💻🍆 Gerry Guinardi')
        st.caption("life moves pretty fast")
        st.caption('[GitHub](https://github.com/rousin1408) [Website](https://www.github.com)')

    with col2:
        st.write('### 👨🏻‍💻🥔 Ho Michael Gunawan')
        st.caption('start from small one and never underestimate the beginning')
        st.caption('[GitHub](https://github.com/mikegun06) [Website](https://mikegun06.github.io/portfolio/)')
    
    with col3:
        st.write('### 👨🏻‍💻🥕 Vito Pramudita Minardi')
        st.caption('wants to sleep early but my code says the otherwise')
        st.caption('[GitHub](https://www.github.com/vitopm) [Website](https://www.vitopm.com)')
    
    st.text('''
    🥦























































































    🥦
    ''')
    colgif, colempty = st.columns([1,2])
    with colgif:
        st.image('asset/video/crying_kid.gif')

if __name__ == '__main__':
    main()