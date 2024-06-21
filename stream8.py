import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import tempfile
import cv2
import numpy as np
from tracker import *
import pandas as pd

def main():
    st.title("PENCACAH LALULINTAS")
    cap = cv2.VideoCapture('potong.mp4')
    tempat = st.empty()
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        st.write(str(i))
        i=i+1
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
