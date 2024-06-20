import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def main():
    st.title("PENCACAH LALULINTAS")
    col1,col2,col3,col4 = st.columns(4)
    file = 'ban.jpg'
    classes = ["ban"]
    img = cv2.imread(str(file))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
