import streamlit as st
import cv2
import numpy as np
import tempfile
cap = cv2.VideoCapture("495.mp4")
st.title("Video Open")
stop = st.button('Stop')
tempat = st.empty()
while cap.isOpened() and not stop :
    ret, frame = cap.read()
    if not ret:
        st.write('selesai')
        break
    
   
    tempat.image(frame,channels='RGB')
    if cv2.waitKey(1) & 0xFF == ord("q")  or stop:
        break
    