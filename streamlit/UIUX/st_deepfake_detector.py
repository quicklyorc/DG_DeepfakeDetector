#rev5-1.chart_hist / tab4(final area) add 

#필요한 모듈 
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import altair as alt
import plotly.express as px 
import cv2
import tempfile
from typing import Tuple, Union
import math
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#csv file load
model1 = pd.read_csv('history/history1.csv', index_col=0)
model2 = pd.read_csv('history/history2.csv', index_col=0)
model3 = pd.read_csv('history/history3.csv', index_col=0)
model4 = pd.read_csv('history/history4.csv', index_col=0)
model5 = pd.read_csv('history/history5.csv', index_col=0)


#CSS클래스 정의
# 폰트 
font_dc = """
    <style>
    @font-face {
    font-family: 'Pretendard-Regular';
    src: url('https://cdn.jsdelivr.net/gh/Project-Noonnu/noonfonts_2107@1.1/Pretendard-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
    }
    </style>
    """

# 배경 이미지 CSS
page_bg_img_first = f"""
    <style>
    .stApp {{
        background-image: url("https://i.postimg.cc/jjVrkT31/Team-Detective-Garfield.png");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: local;
        color: #000000;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
"""

#얼굴 인식 사각형 사이즈 설정
MARGIN = 20  # pixels
ROW_SIZE = 25  # pixels
FONT_SIZE = 5
FONT_THICKNESS = 3
TEXT_COLOR = (0, 255, 0)  # green
TEXT_COLOR_1 = (0, 0, 255)  # red

#얼굴에 사각형 그리기
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(
    image,
    detection_result,
    label 
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    if label =='real': 
      cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
    elif label == 'fake':
      cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR_1, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    if label =='real':
       cv2.putText(annotated_image, label, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS) 
    elif label == 'fake':
       cv2.putText(annotated_image, label, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR_1, FONT_THICKNESS)

  return annotated_image

#최종적인 fake와 real 출력
def classify_data(label_list):
    count_0 = label_list.count(0)
    count_1 = label_list.count(1)
    
    if count_0 > count_1:
        return "real"
    elif count_1 > count_0:
        return "fake"
    else:
        return "equal"
    
# 임시 변수 선언
label_list = []


# 마스크 착용 여부 레이블 및 색상 지정
labels_dict = {1: 'fake', 0: 'real'}
color_dict = {1: (0, 0, 255), 0: (0, 255, 0)}
    
# 모델 불러오기
model = keras.models.load_model('model/crop3-034-0.0432.hdf5')
# model = keras.models.load_model('model/model.tflite')
base_options = python.BaseOptions(model_asset_path='model/detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

def main():
    st.set_page_config(initial_sidebar_state="expanded")
    #sidebar -> button 생성 
    with st.sidebar:
        choice = option_menu("MENU", ["Introduction", "Description", "Detector"],
                            icons=[	'caret-right-fill', 'caret-right-fill', 'caret-right-fill'],
                            menu_icon="card-text", default_index=0,
                            styles={
                                "container": {"padding": "4!important", "background-color": '#FFFFFF'},
                                "icon": {"color": "black", "font-size": "25px"},
                                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": '#FFFFFF'},
                                "nav-link-selected": {"background-color": "#72DDCE", "secondaryBackgroundColor":'#FFFFFF'}})
    

    #[Introduction]Tab 설정 
    if choice == "Introduction":
        st.markdown(page_bg_img_first, unsafe_allow_html=True)
        st.markdown(font_dc+'<div style="font-family: Pretendard-Regular; font-size: 100px; font-weight: 900; color: #fafafa; line-height: 100px; letter-spacing: 5px">Deepfake</div>', unsafe_allow_html=True)
        st.markdown(font_dc+'<div style="font-family: Pretendard-Regular; font-size: 100px; font-weight: 900; color: #fafafa; line-height: 70px; letter-spacing: 5px">Detector</div>', unsafe_allow_html=True)
        st.markdown(font_dc+'<div style="font-family: Pretendard-Regular; font-size: 50px; color: #fafafa; letter-spacing: 2px">Mini Project2</div>', unsafe_allow_html=True)
        st.markdown(font_dc+'<div style="font-family: Pretendard-Regular; font-size: 20px; text-align:center; margin-top: 250px; color: #fafafa;">Team : Detective Garfield</div>', unsafe_allow_html=True)
        

    #[Model-Description]Tab 설정
    elif choice == "Description":
        st.markdown(font_dc+'<div style="font-family: Pretendard-Regular; font-size: 40px; font-weight: 900; color: #04B8C4; line-height: 60px; letter-spacing: 2px">Summary</div>', unsafe_allow_html=True)
  
        col1, col2 = st.columns(2)
        with col1:
            data_th = {'Model_No' : ['1', '2', '3', '4', '5'],
                'Accuracy' : ['1.00000', '0.99900', '0.99900', '0.99450', '0.99178'],
                'F1-score' : ['1.00000', '0.99902', '0.99901', '0.99455', '0.99144'],
                'Recall' : ['1.00000', '0.99902', '1.00000', '1.00000', '0.99775']}
            df = pd.DataFrame(data_th)
            st.dataframe(data_th)
    
        with col2:
            # 숫자열로 변환
            df[['Accuracy', 'F1-score', 'Recall']] = df[['Accuracy', 'F1-score', 'Recall']].apply(pd.to_numeric)
            
            hist_fig = plt.figure(figsize=(8,7))
            hist_ax = hist_fig.add_subplot(111)
            sub_df = df[['Accuracy', 'F1-score', 'Recall']]
            sub_df.plot.bar(alpha = 0.8, ax = hist_ax, title = 'chart', color=('skyblue', 'lightgreen', 'salmon'))
            hist_ax.set_xticklabels(['no1', 'no2', 'no3', 'no4', 'no5'])  # x 축 레이블 설정
            hist_ax.set_ylim(0.98, 1.0)  # y 축 범위 설정
            hist_ax.set_yticks([0.98, 1.0])  # y 축 눈금 설정
            st.pyplot(hist_fig)        
        
        #chart별 tab 분리 
        tab1, tab2, tab3 = st.tabs(['History', 'Confusion matrix', 'Final'])

        #[line_chart]
        with tab1:
            st.subheader('Model1')
            st.line_chart(pd.DataFrame(model1, columns=['loss', 'accuracy', 'val_loss', 'val_accuracy']))
            st.subheader('Model2')
            st.line_chart(pd.DataFrame(model2, columns=['loss', 'accuracy', 'val_loss', 'val_accuracy']))
            st.subheader('Model3')
            st.line_chart(pd.DataFrame(model3, columns=['loss', 'accuracy', 'val_loss', 'val_accuracy']))
            st.subheader('Model4')
            st.line_chart(pd.DataFrame(model4, columns=['loss', 'accuracy', 'val_loss', 'val_accuracy']))
            st.subheader('Model5')
            st.line_chart(pd.DataFrame(model5, columns=['loss', 'accuracy', 'val_loss', 'val_accuracy']))

        #[confusion_matrix]
        with tab2:
            #model1,2의 confusion matrix 
            col3, col4 = st.columns(2)  

            with col3:
                st.subheader('Model1')
                conf_matrix = np.array([[1021, 0],
                                        [0, 979]])
                plt.figure(figsize=(6, 6)) #그림size
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)  #annot: 숫자 표시, fmt=d: 정수표현, cbar: 측면에 나타나는 colorbar, square: 행,열 크기비례
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(plt)
 
            with col4:
                st.subheader('Model2')
                conf_matrix = np.array([[981, 1],
                                        [1, 1017]])
                plt.figure(figsize=(6, 6))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(plt)

            #model3,4의 confusion matrix 
            col5, col6 = st.columns(2)  

            with col5:
                st.subheader('Model3')
                conf_matrix = np.array([[985, 2],
                                        [0, 1013]])
                plt.figure(figsize=(6, 6))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)  
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(plt)
    
            with col6:
                st.subheader('Model4')
                conf_matrix = np.array([[986, 11],
                                        [0, 1003]])
                plt.figure(figsize=(6, 6))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(plt)

            #model5의 confusion matrix 
            col7, col8 = st.columns(2)  

            with col7:
                st.subheader('Model5')
                conf_matrix = np.array([[4331, 60],
                                        [9, 3994]])
                plt.figure(figsize=(6, 6)) 
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)  
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(plt)
 
        with tab3:
            st.header('VGG16 Model')
            st.subheader('History')
            st.image('img/loss.png')
            st.subheader('Confusion matrix')
            conf_matrix = np.array([[1783, 0],
                                    [4, 1773]])
            plt.figure(figsize=(6, 6)) #그림size
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)  #annot: 숫자 표시, fmt=d: 정수표현, cbar: 측면에 나타나는 colorbar, square: 행,열 크기비례
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(plt)
            st.subheader('Occlusion Sensitivity')
            st.text('patch-size=20')
            st.image('img/all_patchsize20.png', use_column_width = True)
            st.text('patch-size=50')
            st.image('img/crop_patchsize50.png', use_column_width = True)
            


    #[Model]Tab 설정 
    elif choice == "Detector":  #가연's back-end area 
        st.markdown(font_dc+'<div style="font-family: Pretendard-Regular; font-size: 40px; font-weight: 900; color: #04B8C4; line-height: 60px; letter-spacing: 2px">Deepfake Detector</div>', unsafe_allow_html=True)
        
        st.text_input('',   placeholder="🎞️ Upload your video 🎞️")
        uploaded_file = st.file_uploader('', type=["mp4", "MOV"])
        
        # 업로드된 파일이 있을 경우 처리
        if uploaded_file is not None:
            # 임시 파일로 저장
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.read())
                
            result = [0.0, 0.0]
            result[0] = 0.0
            
            # 비디오 스트림 읽기
            video_stream = cv2.VideoCapture("temp_video.mp4")
            
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            v_cap = cv2.VideoCapture(tfile.name)
            frameST = st.empty()
            
            # 비디오 writer 객체 생성
            fps = round(video_stream.get(cv2.CAP_PROP_FPS))
            frame_width, frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 비디오 스트림이 열려 있는 동안 반복
            while True:
                #프레임 얼마인지 확인해서 , 
                ret, img = video_stream.read()  # 프레임 읽기
                if ret:
                    # 이미지를 RGB 형식으로 변환
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
                    results = detector.detect(mp_image)

                    if(int(video_stream.get(1)) % fps == 0) :#or video_stream.get(1) == 1:
            
                        x = int(results.detections[0].bounding_box.origin_x)
                        y = int(results.detections[0].bounding_box.origin_y)
                        w = int(results.detections[0].bounding_box.width)
                        h = int(results.detections[0].bounding_box.height)

                        roi = img[y:y+h, x:x+w]
                        resized_roi = cv2.resize(roi,(256, 256))
                        input_image = np.expand_dims(resized_roi, axis=0)
                        result = model.predict(input_image)
                        label_list.append(result[0])
                        
                    image_copy = np.copy(mp_image.numpy_view())
                    label = labels_dict[int(result[0])]
                    annotated_image = visualize(image_copy, results, label) #results -> detection_result
                    frameST.image(annotated_image, channels='BGR', use_column_width=True) #annotated_image
                
                else:
                    break

            # 비디오 스트림 및 VideoWriter 리소스 해제
            video_stream.release()
            result = classify_data(label_list)
            st.subheader("🔍 모델 감지 결과 : {}".format(str(result)))
        

if __name__ == '__main__':
    main()