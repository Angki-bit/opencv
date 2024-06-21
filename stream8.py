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
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        x1 = st.slider('x1',min_value=0,max_value=1000,value=11)
        x4 = st.slider('x4',min_value=0,max_value=1000,value=244)
       
    with col2:
        y1 = st.slider('y1',min_value=0,max_value=1000,value=159)
        y4 = st.slider('y4',min_value=0,max_value=1000,value=427)
      
    with col3:
        x2 = st.slider('x2',min_value=0,max_value=1000,value=61)
        x3 = st.slider('x3',min_value=0,max_value=1000,value=305)
        
    with col4:
        y2 = st.slider('y2',min_value=0,max_value=1000,value=140)
        y3 = st.slider('y3',min_value=0,max_value=1000,value=378)
       
    st.sidebar.title('Setting')
    
    points = np.array([[x1,y1],[x4,y4],[x3,y3],[x2,y2]],np.int32)
    tempat = st.empty()
    tracker = EuclideanDistTracker()
    cap = cv2.VideoCapture('potong.mp4')
    with open('coco.names','r') as f:
        classes = f.read().splitlines()
    caffe = "frozen_inference_graph.pb"
    config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    net = cv2.dnn_DetectionModel(caffe, config_file)
    
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    thres = 0.6
    nms_threshold = 0.6
    tab = st.empty()
    id_nya =[]
    gam =1
    jumlah = 0
    kol1,kol2,kol3,kol4 = st.columns(4)
    motor =0
    mobil =0
    va =0
    with kol1:
        gambar1= st.empty()
        h1 = st.empty()
    with kol2:
        gambar2= st.empty()
        h2 = st.empty()
    with kol3:
        gambar3= st.empty()
        h3 = st.empty()
    with kol4:
        gambar4= st.empty()
        h4 = st.empty()
    jm= st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        dim = (600, 450)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        classIds, confs, bbox = net.detect(frame,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))  
        det = []
        detect = []
        h=0;
       
        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold) 
        if len(classIds) != 0:
           
            for i in indices:
                box = bbox[indices[0]]
                confidence = str(round(confs[indices[0]],2))
                x,y,w,h = box[0],box[1],box[2],box[3]
                det.append([x, y, w, h])
                boxes_ids = tracker.update(det)
                for box_id in boxes_ids:
                   
                    x, y, w, h, id = box_id
                    x1 = int(w/2)
                    y1 = int(h/2)
                    cx = x + x1
                    cy = y + y1
                    x2 = x+w
                    detect.append([cx,cy])
                    result = cv2.pointPolygonTest(points, (int(cx),int(cy)), False)
                    if (result == 1) :
                        #print(gam)
                        roi = frame[int(y)-20:int(y)+int(h)+10,int(x):int(x)+int(w)+10]
                        cv2.putText(frame, str(id), (cx, cy),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)
                        if (len(id_nya)==0):
                            id_nya.append(id)
                            gambar1.image(roi)
                            tf = tempfile.NamedTemporaryFile(suffix=".jpg",prefix="Compare_")
   
                        gam = gam + 1
                        if (gam == 5):
                            gam = 1
   
                            
        if len(id_nya) == 100 :
            id_nya =[]
        cv2.polylines(frame,[points],True,[0,255,0],thickness=1)
        if not ret:
            st.write('selesai')
            break
           
        tempat.image(frame,channels='RGB')
        df = pd.DataFrame([[motor,mobil,va,0,0,0,0]], columns=("I","II/III/IV","V a","V b","VI a","VI b","VIIa/b/c"))
        tab.table(df)
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
