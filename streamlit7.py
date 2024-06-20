import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

def main():
    st.title("PENCACAH LALULINTAS")
    st.image('ban.JPG')

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
