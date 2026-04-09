import cv2
import numpy as np
import os
import glob
import time
import zipfile
import io
from flask import Flask, render_template, request, url_for, send_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

class UniversalFaceBlur:
    def __init__(self):
        # 💡 핵심: 이제 파일을 다운로드하지 않고, 깃허브에 올린 파일을 바로 읽습니다.
        self.proto_path = "deploy.prototxt"
        self.model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(self.proto_path) or not os.path.exists(self.model_path):
            raise Exception("AI 모델 파일이 서버에 없습니다. 깃허브 업로드를 확인해주세요.")
            
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

    def is_real_human(self, face_img, confidence):
        if face_img.size == 0: return False
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        if std_dev < 12: return False
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:,:,1])
        if avg_saturation > 180: return False
        h, w = face_img.shape[:2]
        if w * h > 10000 and confidence < 0.40: return False
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
            effect_roi = cv2.GaussianBlur(roi, (max(15, k), max(15, k)), 0)
        elif option == 'heart':
            self.draw_3d_heart(img, x + center[0], y + center[1], radius)
            return
        img[y:y+h, x:x+w] = np.where(mask == 255, effect_roi, roi)

    def process_web_image(self, input_path, output_path, option):
        img = cv2.imread(input_path)
        if img is None: return False
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.25: 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY, endX, endY = max(0, startX), max(0, startY), min(w, endX), min(h, endY)
                if endX - startX > 5 and endY - startY > 5:
                    if self.is_real_human(img[startY:endY, startX:endX], confidence):
                        self.apply_effect(img, startX, startY, endX-startX, endY-startY, option)
        cv2.imwrite(output_path, img)
        return True

face_blurrer = UniversalFaceBlur()

@app.route('/')
def index():
    return render_template('index.html', result_images=None)

@app.route('/upload', methods=['POST'])
def upload_files():
    # 폴더 비우기
    for f in glob.glob(os.path.join(app.config['RESULT_FOLDER'], '*')): os.remove(f)
    uploaded_files = request.files.getlist("files")
    selected_option = request.form.get("option")
    result_filenames = []
    for i, file in enumerate(uploaded_files[:10]):
        if file.filename != '':
            ext = file.filename.split('.')[-1]
            filename = f"photo_{int(time.time())}_{i}.{ext}"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_filename = f"done_{filename}"
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
    app.run(host='0.0.0.0', port=5000)
