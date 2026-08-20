import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_landmarks(img):
    """顔の特徴点（ランドマーク）を取得"""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        h, w, _ = img.shape
        return np.array([(int(pt.x * w), int(pt.y * h)) for pt in results.multi_face_landmarks[0].landmark])

def generate_average_face(img1, img2):
    """画像をリサイズしてアルファブレンディング（50%ずつ合成）"""
    h, w = 600, 600
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    
    # 2つの画像を均等に合成
    blended = cv2.addWeighted(img1_resized, 0.5, img2_resized, 0.5, 0)
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
        result = generate_average_face(img1, img2)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="生成された平均顔", use_container_width=True)