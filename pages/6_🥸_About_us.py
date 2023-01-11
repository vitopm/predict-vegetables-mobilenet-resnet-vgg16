import streamlit as st

def main():
    st.image('./asset/images/vege-background.jpg')
    st.title("🥸 About us")
    st.write('🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜🍅🍆🥒🥕🥔🥜')
    st.caption('Made by students, for student, with the help of internet')

    st.write('----')
    
    st.write('#### 🍆 Gerry Guinardi')
    st.caption("life moves pretty fast")
    st.write('#### 🥔 Michael Gunawan')
    st.caption('start from small one and never underestimate the beginning')
    st.write('#### 🥕 Vito Pramudita Minardi')
    st.caption('Wants to sleep early but my code says the otherwise')
    
if __name__ == '__main__':
    main()