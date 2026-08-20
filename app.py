import os
import urllib.request
import cv2
import numpy as np
import streamlit as st

# 安定したCDNサーバーから設定ファイルを取得
CASCADE_URL = "https://cdn.jsdelivr.net/gh/opencv/opencv@4.x/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

def get_cascade():
    # ファイルが存在しない、または破損している(XMLでない)場合は再取得
    if not os.path.exists(CASCADE_PATH) or os.path.getsize(CASCADE_PATH) < 10000:
        try:
            req = urllib.request.Request(CASCADE_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read()
                if b"<?xml" in content:
                    with open(CASCADE_PATH, 'wb') as f:
                        f.write(content)
        except Exception:
            pass

    cascade = cv2.CascadeClassifier()
    if os.path.exists(CASCADE_PATH):
        cascade.load(CASCADE_PATH)
    return cascade

def crop_and_blend_faces(img1, img2):
    cascade = get_cascade()
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    faces1 = cascade.detectMultiScale(gray1, 1.1, 5, minSize=(100, 100)) if not cascade.empty() else []
    faces2 = cascade.detectMultiScale(gray2, 1.1, 5, minSize=(100, 100)) if not cascade.empty() else []
    
    # 顔が検出された場合は顔を中心に、未検出の場合は写真の中央60%をカット
    def get_face_or_center(img, faces):
        h, w = img.shape[:2]
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            return img[y:y+fh, x:x+fw]
        else:
            cy, cx = h // 2, w // 2
            size = int(min(h, w) * 0.6 // 2)
            return img[max(0, cy-size):min(h, cy+size), max(0, cx-size):min(w, cx+size)]

    face1 = cv2.resize(get_face_or_center(img1, faces1), (500, 500))
    face2 = cv2.resize(get_face_or_center(img2, faces2), (500, 500))
    
    if len(faces1) == 0 or len(faces2) == 0:
        st.info("※一部の画像で顔の位置を自動特定できなかったため、中心部を基準に切り抜いて合成しました。")
        
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
