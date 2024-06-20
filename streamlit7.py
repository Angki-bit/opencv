import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def main():
    st.title("PENCACAH LALULINTAS")
    img1 = 'ban.JPG'
    img = cv2.imread(img1)
    st.image(img,channels='RGB')

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
