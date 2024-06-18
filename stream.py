import streamlit as st
import tempfile
import cv2
import numpy as np
from tracker import *


def putar(nama_file,x1,y1,x2,y2,x3,y3,x4,y4):
    points = np.array([[x1,y1],[x4,y4],[x3,y3],[x2,y2]],np.int32)
    tempat = st.empty()
  
    cap = cv2.VideoCapture(str(nama_file))
    st.write(x1)
    gam = st.empty()
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        gambar1 = st.empty()
        ket1 = st.empty()
    with col2:
        gambar2 = st.empty()
        ket2 = st.empty()
    with col3:
        gambar3 = st.empty()
        ket3 = st.empty()
    with col4:  
        gambar4 = st.empty()
        ket4 = st.empty()
        
    tracker = EuclideanDistTracker()
    with open('coco.names','r') as f:
        classNames = f.read().splitlines()

    caffe = "frozen_inference_graph.pb"
    config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    net = cv2.dnn_DetectionModel(caffe, config_file)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    thres = 0.6
    nms_threshold = 0.6
    font = cv2.FONT_HERSHEY_PLAIN
    Colors = np.random.uniform(0, 255, size=(len(classNames), 3))    
    dim = (700, 500)
    c = 1
    id_nya =[]
    z=1
    
    while cap.isOpened():
        ret, img = cap.read()
        #cv2.imwrite('tes.jpg', img)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
    
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))  
        det = []
        detect = []
        h=0;
        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)   
        #cv2.polylines(img, [points], True, (0, 255, 0), thickness=1)
        #cv2.imwrite('capture/'+str(c)+'.jpg', img)
        
        
        if len(classIds) != 0:
            
            for i in indices:
                kelas = classIds[indices[0]-1]
                nama = classNames[kelas-1]
                box = bbox[indices[0]]
                confidence = str(round(confs[indices[0]],2))
                x,y,w,h = box[0],box[1],box[2],box[3]
                det.append([x, y, w, h])
                boxes_ids = tracker.update(det)
               
                
                for box_id in boxes_ids:
                    x, y, w, h, id = box_id

                    x1 = int(w//2)
                    y1 = int(h//2)
                    cx = x + x1
                    cy = y + y1
                    x2 = x+w
                    detect.append([cx,cy])
                    fs = int(y)-10
                    bs= int(y)+int(h)+10
                    cs= int(x)
                    ds = int(x)+int(w)+60
                    result = cv2.pointPolygonTest(points, (int(cx),int(cy)), False)
                    if (result == 1) :
                        j_id = len(id_nya)
                        if j_id == 0:
                            id_nya.append(id)
                            roi = img[fs:bs,cs:ds]
                            gambar1.image(roi) 
                            ket1.write(str(z))
                           
                            z=z+1
                        if j_id >0:
                            if id not in id_nya:
                                id_nya.append(id)
                                roi = img[fs:bs,cs:ds]
                                
                                gam.write(id)
                                cv2.putText(img, str(id), (cx, cy),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)
                                if z==1:
                                    gambar1.image(roi) 
                                    ket1.write(str(z))
                                if z==2:
                                    gambar2.image(roi) 
                                    ket2.write(str(z))
                                if z==3:
                                    gambar3.image(roi) 
                                    ket3.write(str(z))
                                if z==4:
                                    gambar4.image(roi) 
                                    ket4.write(str(z))
                                print(z)
                                z=z+1
                                
                                if z == 5:
                                    z=1
                    else:
                        roi = img[120:180,110:190]
                    
                    
                    
                    
                    
        if not ret:
            st.write('selesai')
            break
       
        tempat.image(img,channels='RGB')
        c= c+1
        
        
    
def main():
    st.title("PENCACAH LALULINTAS")
    st.sidebar.title('Setting')
    
    st.markdown(
        '''
        <style>
        [data-testid="stSidebar"][aria-expaded="true"] > div:first-child{width:400px;}
        [data-testid="stSidebar"][aria-expaded="false"] > div:first-child{width:400px;margin-left:-400px}
        </style>
        ''',unsafe_allow_html=True,
    )
    st.sidebar.markdown('---')
    x1 = st.sidebar.slider('x1',min_value=0,max_value=1000,value=37)
    y1 = st.sidebar.slider('y1',min_value=0,max_value=1000,value=221)
    
    x2 = st.sidebar.slider('x2',min_value=0,max_value=1000,value=114)
    y2 = st.sidebar.slider('y2',min_value=0,max_value=1000,value=201)
    
    x3 = st.sidebar.slider('x3',min_value=0,max_value=1000,value=362)
    y3 = st.sidebar.slider('y3',min_value=0,max_value=1000,value=409)
    
    x4 = st.sidebar.slider('x4',min_value=0,max_value=1000,value=285)
    y4 = st.sidebar.slider('y4',min_value=0,max_value=1000,value=463)
    st.sidebar.markdown('---')
    putar('http://localhost/lhr/video/128k11/495.mp4',x1,y1,x2,y2,x3,y3,x4,y4)
   
    video_file_buffer = st.sidebar.file_uploader("Upload Video",type=["mp4","avi"])
    DEMO_VIDEO = '495.mp4' 
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) 
    
    if not video_file_buffer:
        vid=cv2.VideoCapture(DEMO_VIDEO)
        tffile.name = DEMO_VIDEO
        dem_vid = open(tffile.name,'rb')
        demo_bytes = dem_vid.read()
        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
    else:
        tffile.write(video_file_buffer.read())
        dem_vid = open(tffile.name,'rb')
        demo_bytes = dem_vid.read()
        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
        #putar(tffile.name,x1,y1,x2,y2,x3,y3,x4,y4)
        
     
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

