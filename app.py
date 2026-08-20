import os
import urllib.request
import cv2
import numpy as np
import streamlit as st

CASCADE_PATH = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

def ensure_cascade_file():
    """設定ファイルを安全に用意する関数"""
    if not os.path.exists(CASCADE_PATH) or os.path.getsize(CASCADE_PATH) < 10000:
        try:
            req = urllib.request.Request(CASCADE_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = resp.read()
                if len(data) > 10000:
                    with open(CASCADE_PATH, 'wb') as f:
                        f.write(data)
        except Exception:
            pass

def extract_face_or_center(img):
    """顔を切り抜く関数（エラー時は中央部を切り抜き）"""
    h, w = img.shape[:2]
    
    try:
        ensure_cascade_file()
        if os.path.exists(CASCADE_PATH):
            cascade = cv2.CascadeClassifier(CASCADE_PATH)
            if not cascade.empty():
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                    return img[y:y+fh, x:x+fw], True
    except Exception:
        pass
        
    # 顔認識失敗・エラー時は安全に中央をトリミング
    cy, cx = h // 2, w // 2
    size = int(min(h, w) * 0.6 // 2)
    return img[max(0, cy-size):min(h, cy+size), max(0, cx-size):min(w, cx+size)], False

def blend_faces(img1, img2):
    face1, ok1 = extract_face_or_center(img1)
    face2, ok2 = extract_face_or_center(img2)
    
    if not (ok1 and ok2):
        st.info("※顔の位置を自動特定できなかったため、写真の中心部を基準に切り抜いて合成しました。")
        
    face1_resized = cv2.resize(face1, (500, 500))
    face2_resized = cv2.resize(face2, (500, 500))
    
    return cv2.addWeighted(face1_resized, 0.5, face2_resized, 0.5, 0)

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
        result = blend_faces(img1, img2)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="生成された平均顔", use_container_width=True)
