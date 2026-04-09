import cv2
import numpy as np
import os
import glob
import time
import zipfile
import io
from flask import Flask, render_template, request, url_for, send_file

app = Flask(__name__)

# [해결] 클라우드 서버의 절대 경로를 확실하게 잡습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'results')

# 폴더 자동 생성 (에러 방지)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

class UniversalFaceBlur:
    def __init__(self):
        # [해결] 모델 파일 경로를 절대 경로로 지정하여 '파일 없음' 오류 차단
        self.proto_path = os.path.join(BASE_DIR, "deploy.prototxt")
        self.model_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        
        if not os.path.exists(self.proto_path) or not os.path.exists(self.model_path):
            self.net = None
            print("⚠️ 모델 파일이 없습니다. 깃허브 업로드를 확인하세요.")
            return

        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

    def apply_effect(self, img, x, y, w, h, option):
        ih, iw = img.shape[:2]
        pad_w, pad_h = int(w * 0.30), int(h * 0.30)
        nx, ny = max(0, x - pad_w), max(0, y - pad_h)
        nw, nh = min(iw - nx, w + pad_w * 2), min(ih - ny, h + pad_h * 2)
        
        roi = img[ny:ny+nh, nx:nx+nw]
        if roi.size == 0: return
        
        effect_roi = roi.copy()
        mask = np.zeros((nh, nw, 3), dtype=np.uint8)
        cv2.circle(mask, (nw//2, nh//2), int(max(nw, nh)*0.55), (255, 255, 255), -1)

        if option == 'mosaic':
            b = max(4, nw // 30)
            small = cv2.resize(roi, (b, b), interpolation=cv2.INTER_LINEAR)
            effect_roi = cv2.resize(small, (nw, nh), interpolation=cv2.INTER_NEAREST)
        elif option == 'blur':
            k = (int(nw / 1.5) // 2 * 2) + 1 # 반드시 홀수
            effect_roi = cv2.GaussianBlur(roi, (max(15, k), max(15, k)), 0)
        
        img[ny:ny+nh, nx:nx+nw] = np.where(mask == 255, effect_roi, roi)

    def process_web_image(self, input_path, output_path, option):
        if self.net is None: return False
        img = cv2.imread(input_path)
        if img is None: return False
        
        h, w = img.shape[:2]
        # [해결] 메모리 부족 방지를 위해 분석 해상도를 300x300으로 최적화 (502 에러 주원인)
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] > 0.25:
                box = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype("int")
                self.apply_effect(img, box[0], box[1], box[2]-box[0], box[3]-box[1], option)
        
        cv2.imwrite(output_path, img)
        return True

face_blurrer = UniversalFaceBlur()

@app.route('/')
def index():
    return render_template('index.html', result_images=None)

@app.route('/upload', methods=['POST'])
def upload_files():
    # 처리 전 이전 파일 싹 지우기 (서버 용량 관리)
    for f in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        try: os.remove(f)
        except: pass
        
    uploaded_files = request.files.getlist("files")
    option = request.form.get("option")
    results = []
    
    for i, file in enumerate(uploaded_files[:5]): # 무료 서버 사양을 고려해 한 번에 5장으로 제한
        if file and file.filename:
            fname = f"img_{int(time.time())}_{i}.jpg"
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

# 기존 맨 밑의 if __name__ == '__main__': 부분을 아래처럼 아주 심플하게 바꿉니다.
if __name__ == '__main__':
    app.run()
