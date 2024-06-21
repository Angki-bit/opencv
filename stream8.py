import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import tempfile
import cv2
import numpy as np
from tracker import *
import pandas as pd
def analisis(roi):
    hasil = st.empty
       
    img = cv2.imread(roi)
    classes = ["motor","lainnya","mobil","pickup","truk gandeng","bus kecil","truk sedang","bus besar","tiga sumbu","truk ringan","trailer"]
    net = cv2.dnn.readNetFromONNX("mobil.onnx")
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=[640, 640], mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]
    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640
    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.2:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            
            if classes_score[ind] > 0.2:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx-w/2)*x_scale)
                    y1 = int((cy-h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1, y1, width, height])
                    boxes.append(box)
               
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
    jenis =[]
    for i in indices:
        x1, y1, w, h = boxes[i]
        label = classes[classes_ids[i]]
        jenis.append(label)
        
    if len(jenis)==0:
        jenis.append('motor')
        hasil = jenis[0]
    else:
        jm = len(jenis)
        hasil = jenis[jm-1] 
        
    return hasil     
       
        
    
     
    

def main():
    st.title("PENCACAH LALULINTAS")
    
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
