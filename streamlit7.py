import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2


def main():
    st.title("PENCACAH LALULINTAS")
    img1 = 'ban.JPG'
    img = cv2.imread(img1)
    st.image(img,channels='RGB')
    net = cv2.dnn.readNetFromONNX('mobil.onnx')
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=[640, 640], mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
