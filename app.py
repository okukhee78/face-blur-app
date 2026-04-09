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

# [해결] 클라우드 환경에서 SSL 보안 에러 방지
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# [해결] 상대 경로 대신 절대 경로를 사용하여 클라우드 서버의 경로 혼선을 방지합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'results')

# 폴더 자동 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

class UniversalFaceBlur:
    def __init__(self):
        # 💡 [중요] 깃허브에 올린 모델 파일 이름을 절대 경로로 지정합니다.
        self.proto_path = os.path.join(BASE_DIR, "deploy.prototxt")
        self.model_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # 파일이 없을 때만 다운로드 시도
        self.download_models()
        
        # AI 네트워크 로드
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

    def download_models(self):
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        if not os.path.exists(self.proto_path): urllib.request.urlretrieve(proto_url, self.proto_path)
        if not os.path.exists(self.model_path): urllib.request.urlretrieve(model_url, self.model_path)

    def is_real_human(self, face_img, confidence):
        """인식률이 가장 좋았던 2차 검증 필터"""
        if face_img.size == 0: return False
        
        try:
            # 1. 질감 검사
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(gray)
            if std_dev < 12: return False
                
            # 2. 채도 검사
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:,:,1])
            if avg_saturation > 180: return False
                
            # 3. 크기 대비 신뢰도 검사
            h, w = face_img.shape[:2]
            if w * h > 10000 and confidence < 0.40: return False
        except:
            return False
            
        return True

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
        
        # 가림 구역 확대 (인식률 좋았던 설정 적용)
        pad_w, pad_h = int(w * 0.30), int(h * 0.30)
        x, y = max(0, x - pad_w), max(0, y - pad_h)
        w, h = min(iw - x, w + pad_w * 2), min(ih - y, h + pad_h * 2)
        if w <= 0 or h <= 0: return

        roi = img[y:y+h, x:x+w]
        if roi.size == 0: return
        
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
            effect_roi = cv2.GaussianBlur(roi, (max(15, k), max(15, k)), 0)
        elif option == 'heart':
            self.draw_3d_heart(img, x + center[0], y + center[1], radius)
            return

        img[y:y+h, x:x+w] = np.where(mask == 255, effect_roi, roi)

    def process_web_image(self, input_path, output_path, option):
        img = cv2.imread(input_path)
        if img is None: return False

        h, w = img.shape[:2]
        # 인식률 향상을 위한 대비 보정(CLAHE)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])
        enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # DNN 입력 (고해상도 설정)
        blob = cv2.dnn.blobFromImage(enhanced_img, 1.0, (1200, 1200), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.20: # 의심 기준 20% 유지
                box = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype("int")
                startX, startY, endX, endY = max(0,box[0]), max(0,box[1]), min(w,box[2]), min(h,box[3])
                
                face_w, face_h = endX - startX, endY - startY
                if face_w > 5 and face_h > 5:
                    if self.is_real_human(img[startY:endY, startX:endX], confidence):
                        self.apply_effect(img, startX, startY, face_w, face_h, option=option)
        
        cv2.imwrite(output_path, img)
        return True

face_blurrer = UniversalFaceBlur()

@app.route('/')
def index():
    return render_template('index.html', result_images=None)

@app.route('/upload', methods=['POST'])
def upload_files():
    # [해결] 파일 삭제 시 에러가 나도 서버가 뻗지 않도록 예외처리 강화
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
        if file and file.filename:
            # [해결] 한글 파일명 오류 방지를 위한 안전한 이름 생성
            ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
            safe_filename = f"photo_{int(time.time())}_{i}.{ext}"
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            output_filename = f"done_{safe_filename}"
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            
            file.save(input_path) 
            if face_blurrer.process_web_image(input_path, output_path, selected_option):
                result_filenames.append(output_filename)

    return render_template('index.html', result_images=result_filenames)

@app.route('/download_all')
def download_all():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for f in glob.glob(os.path.join(app.config['RESULT_FOLDER'], 'done_*')):
            zf.write(f, os.path.basename(f))
    memory_file.seek(0)
    return send_file(memory_file, download_name='blurred_photos.zip', as_attachment=True)

if __name__ == '__main__':
    # [해결] 렌더 환경에서는 환경변수 PORT를 반드시 읽어와야 접속 차단(502)이 안 납니다.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
