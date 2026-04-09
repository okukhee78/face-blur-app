import cv2
import numpy as np
import urllib.request
import os
import glob
import ssl
import math
import time
import zipfile
import io
from flask import Flask, render_template, request, url_for, send_file

# 다운로드 보안 에러 방지
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

class UniversalFaceBlur:
    def __init__(self):
        self.proto_path = "deploy.prototxt"
        self.model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.download_models()
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

    def download_models(self):
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        if not os.path.exists(self.proto_path): urllib.request.urlretrieve(proto_url, self.proto_path)
        if not os.path.exists(self.model_path): urllib.request.urlretrieve(model_url, self.model_path)

    # 💡 핵심: 진짜 사람인지 검증하는 2차 필터 함수 추가!
    def is_real_human(self, face_img, confidence):
        if face_img.size == 0: return False
        
        # 1. 질감 검사: 그림이나 만화는 색이 단조롭습니다.
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        if std_dev < 12: # 질감이 너무 밋밋하면 그림/만화로 간주 (제외)
            return False
            
        # 2. 채도 검사: 인쇄물이나 그림 특유의 쨍한 형광색 걸러내기
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:,:,1])
        if avg_saturation > 180: # 채도가 비정상적으로 높으면 제외
            return False
            
        # 3. 크기 대비 신뢰도 검사: 얼굴이 화면에 크게 잡혔는데 신뢰도가 낮으면 플라스틱 인형일 확률 높음
        h, w = face_img.shape[:2]
        face_area = w * h
        if face_area > 10000 and confidence < 0.40:
            return False
            
        return True # 이 모든 시험을 통과해야 '진짜 사람'으로 인정

    def draw_3d_heart(self, img, cx, cy, size):
        t = np.linspace(0, 2 * np.pi, 100)
        x = 16 * (np.sin(t) ** 3)
        y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        scale = size / 35.0
        x = cx + x * scale
        y = cy + y * scale - (size * 0.1)
        points = np.column_stack((x, y)).astype(np.int32)
        cv2.fillPoly(img, [points], (60, 60, 235)) 
        highlight_x, highlight_y = cx - int(size * 0.25), cy - int(size * 0.2)
        cv2.ellipse(img, (highlight_x, highlight_y), (int(size * 0.12), int(size * 0.06)), -30, 0, 360, (255, 255, 255), -1)

    def apply_effect(self, img, x, y, w, h, option):
        ih, iw = img.shape[:2]
        
        pad_w, pad_h = int(w * 0.30), int(h * 0.30)
        x, y = max(0, x - pad_w), max(0, y - pad_h)
        w, h = min(iw - x, w + pad_w * 2), min(ih - y, h + pad_h * 2)
        if w <= 0 or h <= 0: return

        roi = img[y:y+h, x:x+w]
        effect_roi = roi.copy()
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        center = (w // 2, h // 2)
        
        radius = int(max(w, h) * 0.55)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

        if option == 'mosaic':
            blocks = max(4, w // 30) 
            small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
            effect_roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif option == 'blur':
            k = int(w / 1.5)
            k = k + 1 if k % 2 == 0 else k 
            k = max(15, k)
            effect_roi = cv2.GaussianBlur(roi, (k, k), 0)
            effect_roi = cv2.GaussianBlur(effect_roi, (k, k), 0) 
        elif option == 'heart':
            self.draw_3d_heart(img, x + center[0], y + center[1], radius)
            return

        blended = np.where(mask == 255, effect_roi, roi)
        img[y:y+h, x:x+w] = blended

    def process_web_image(self, input_path, output_path, option):
        img = cv2.imread(input_path)
        if img is None: return False

        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])
        enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        blob = cv2.dnn.blobFromImage(enhanced_img, 1.0, (1200, 1200), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # 흐릿한 사람을 잡기 위해 여전히 의심 기준은 20%로 낮게 유지합니다.
            if confidence > 0.20: 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # 얼굴 박스가 사진 밖으로 나가지 않게 보호
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                face_w, face_h = endX - startX, endY - startY
                
                if face_w > 5 and face_h > 5:
                    # 얼굴 영역을 오려냅니다.
                    face_roi = img[startY:endY, startX:endX]
                    
                    # 🎯 2차 검증 통과 시에만 블러 처리 적용!
                    if self.is_real_human(face_roi, confidence):
                        self.apply_effect(img, startX, startY, face_w, face_h, option=option)
        
        cv2.imwrite(output_path, img)
        return True

face_blurrer = UniversalFaceBlur()

@app.route('/')
def index():
    return render_template('index.html', result_images=None)

@app.route('/upload', methods=['POST'])
def upload_files():
    for f in glob.glob(os.path.join(app.config['RESULT_FOLDER'], '*')): 
        try: os.remove(f)
        except: pass
    for f in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')): 
        try: os.remove(f)
        except: pass

    uploaded_files = request.files.getlist("files")
    selected_option = request.form.get("option")
    
    result_filenames = []
    
    for i, file in enumerate(uploaded_files[:10]): 
        if file.filename != '':
            ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
            safe_filename = f"photo_{int(time.time())}_{i}.{ext}"
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            output_filename = f"done_{safe_filename}"
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            
            file.save(input_path) 
            success = face_blurrer.process_web_image(input_path, output_path, selected_option) 
            
            if success:
                result_filenames.append(output_filename)

    return render_template('index.html', result_images=result_filenames)

@app.route('/download_all')
def download_all():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for f in glob.glob(os.path.join(app.config['RESULT_FOLDER'], 'done_*')):
            filename = os.path.basename(f)
            zf.write(f, filename)
    memory_file.seek(0)
    return send_file(memory_file, download_name='blurred_photos.zip', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
