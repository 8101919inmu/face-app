import cv2
import numpy as np
import streamlit as st

# OpenCV標準の顔検出AI
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_and_blend_faces(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 顔の位置を自動検出
    faces1 = face_cascade.detectMultiScale(gray1, 1.1, 5, minSize=(100, 100))
    faces2 = face_cascade.detectMultiScale(gray2, 1.1, 5, minSize=(100, 100))
    
    # 顔が検出できない場合のフォールバック
    if len(faces1) == 0 or len(faces2) == 0:
        st.warning("顔を上手く検出できませんでした。正面を向いた明らかな写真でお試しください。")
        return cv2.addWeighted(cv2.resize(img1, (500, 500)), 0.5, cv2.resize(img2, (500, 500)), 0.5, 0)
    
    # 最も大きく映っている顔をそれぞれ抽出
    x1, y1, w1, h1 = max(faces1, key=lambda b: b[2] * b[3])
    x2, y2, w2, h2 = max(faces2, key=lambda b: b[2] * b[3])
    
    # 顔のパーツ位置を合わせるため、顔の周りを切り出して同じサイズ(500x500)に統一
    face1 = cv2.resize(img1[y1:y1+h1, x1:x1+w1], (500, 500))
    face2 = cv2.resize(img2[y2:y2+h2, x2:x2+w2], (500, 500))
    
    # 位置が揃った顔同士を合成
    blended = cv2.addWeighted(face1, 0.5, face2, 0.5, 0)
    return blended

st.title("平均顔生成システム")

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("1枚目の顔写真", type=["jpg", "png", "jpeg"])
with col2:
    file2 = st.file_uploader("2枚目の顔写真", type=["jpg", "png", "jpeg"])

if file1 and file2:
    img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

    if st.button("平均顔を生成する"):
        result = crop_and_blend_faces(img1, img2)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="生成された平均顔", use_container_width=True)
