import os
import urllib.request
import cv2
import numpy as np
import streamlit as st

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# アクセスブロックを防ぎながらAI学習用ファイルをダウンロード
if not os.path.exists(CASCADE_PATH) or os.path.getsize(CASCADE_PATH) == 0:
    req = urllib.request.Request(CASCADE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(CASCADE_PATH, 'wb') as out_file:
        out_file.write(response.read())

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(CASCADE_PATH)

face_cascade = load_cascade()

def crop_and_blend_faces(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    faces1 = face_cascade.detectMultiScale(gray1, 1.1, 5, minSize=(100, 100))
    faces2 = face_cascade.detectMultiScale(gray2, 1.1, 5, minSize=(100, 100))
    
    # 顔が検出できない場合は全体をリサイズして重ね合わせ
    if len(faces1) == 0 or len(faces2) == 0:
        st.warning("顔を検出できませんでした。写真全体を切り抜いて合成します。")
        return cv2.addWeighted(cv2.resize(img1, (500, 500)), 0.5, cv2.resize(img2, (500, 500)), 0.5, 0)
    
    x1, y1, w1, h1 = max(faces1, key=lambda b: b[2] * b[3])
    x2, y2, w2, h2 = max(faces2, key=lambda b: b[2] * b[3])
    
    face1 = cv2.resize(img1[y1:y1+h1, x1:x1+w1], (500, 500))
    face2 = cv2.resize(img2[y2:y2+h2, x2:x2+w2], (500, 500))
    
    return cv2.addWeighted(face1, 0.5, face2, 0.5, 0)

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
