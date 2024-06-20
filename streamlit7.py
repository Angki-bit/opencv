import streamlit as st
import numpy as np
import pandas as pd
from tracker import *
import tempfile
import opencv as cv2

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

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
