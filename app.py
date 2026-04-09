import cv2
import numpy as np
import os
import glob
import time
import zipfile
import io
from flask import Flask, render_template, request, url_for, send_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'results')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

class UniversalFaceBlur:
    def __init__(self):
        self.proto_path = os.path.join(BASE_DIR, "deploy.prototxt")
        self.model_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        if not os.path.exists(self.proto_path) or not os.path.exists(self.model_path):
            self.net = None
            return
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

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
        nx, ny = max(0, x - pad_w), max(0, y - pad_h)
        nw, nh = min(iw - nx, w + pad_w * 2), min(ih - ny, h + pad_h * 2)
        
        roi = img[ny:ny+nh, nx:nx+nw]
        if roi.size == 0: return
        
        effect_roi = roi.copy()
        mask = np.zeros((nh, nw, 3), dtype=np.uint8)
        radius = int(max(nw, nh) * 0.55)
        cv2.circle(mask, (nw//2, nh//2), radius, (255, 255, 255), -1)

        # 🎯 핵심: 옵션 값이 대문자든 소문자든, 찰떡같이 알아듣도록 강력하게 세팅!
        option_str = str(option).lower()
        
        if 'blur' in option_str: # 'blur' 글자가 있으면 무조건 흐림 처리
            k = (int(nw / 1.5) // 2 * 2) + 1 
            effect_roi = cv2.GaussianBlur(roi, (max(15, k), max(15, k)), 0)
        elif 'heart' in option_str: # 'heart' 글자가 있으면 무조건 하트 처리
            self.draw_3d_heart(img, nx + nw//2, ny + nh//2, radius)
            return
        else: # 그 외 모든 경우는 모자이크
            b = max(4, nw // 30)
            small = cv2.resize(roi, (b, b), interpolation=cv2.INTER_LINEAR)
            effect_roi = cv2.resize(small, (nw, nh), interpolation=cv2.INTER_NEAREST)
        
        img[ny:ny+nh, nx:nx+nw] = np.where(mask == 255, effect_roi, roi)

    def process_web_image(self, input_path, output_path, option):
        if self.net is None: return False
        img = cv2.imread(input_path)
        if img is None: return False
        
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])
        enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        blob = cv2.dnn.blobFromImage(enhanced_img, 1.0, (600, 600), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.15: 
                box = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype("int")
                startX, startY, endX, endY = max(0,box[0]), max(0,box[1]), min(w,box[2]), min(h,box[3])
                
                face_w, face_h = endX - startX, endY - startY
                if face_w > 5 and face_h > 5:
                    self.apply_effect(img, startX, startY, face_w, face_h, option)
        
        cv2.imwrite(output_path, img)
        return True

face_blurrer = UniversalFaceBlur()

@app.route('/')
def index():
    return render_template('index.html', result_images=None)

@app.route('/upload', methods=['POST'])
def upload_files():
    # 🎯 핵심: 방금 전까지 테스트하며 만들어졌던 옛날 결과물들을 깨끗하게 지웁니다!
    for f in glob.glob(os.path.join(app.config['RESULT_FOLDER'], '*')):
        try: os.remove(f)
        except: pass
    for f in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        try: os.remove(f)
        except: pass
        
    uploaded_files = request.files.getlist("files")
    option = request.form.get("option", "mosaic") # 폼에서 넘어온 옵션 받기
    results = []
    
    for i, file in enumerate(uploaded_files[:5]): 
        if file and file.filename:
            ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
            fname = f"img_{int(time.time())}_{i}.{ext}"
            in_p = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            out_p = os.path.join(app.config['RESULT_FOLDER'], f"done_{fname}")
            file.save(in_p)
            if face_blurrer.process_web_image(in_p, out_p, option):
                results.append(f"done_{fname}")
                
    return render_template('index.html', result_images=results)

@app.route('/download_all')
def download_all():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for f in glob.glob(os.path.join(app.config['RESULT_FOLDER'], 'done_*')):
            zf.write(f, os.path.basename(f))
    memory_file.seek(0)
    return send_file(memory_file, download_name='photos.zip', as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
