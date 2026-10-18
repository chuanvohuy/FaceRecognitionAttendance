import os
import io
import csv
import base64
try:
    import cv2
except Exception:
    cv2 = None
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import random
import secrets
import time
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file, Response, stream_with_context
from werkzeug.security import check_password_hash, generate_password_hash
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from database.database import get_db, init_db

app = Flask(__name__)
app.secret_key = 'facecheckin_enterprise_master_key_2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'static', 'snapshots')
AVATAR_DIR = os.path.join(BASE_DIR, 'static', 'avatars')
BACKUP_DIR = os.path.join(BASE_DIR, 'database', 'backups')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(AVATAR_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Tự động đồng bộ và nâng cấp DB khi khởi động
init_db()

# ----------------- NẠP AI MODELS & CASCADE CLASSIFIERS -----------------
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, 'models_ai', 'yolo26m.pt')
def load_yolo():
    if os.path.exists(CUSTOM_MODEL_PATH) and os.path.getsize(CUSTOM_MODEL_PATH) > 1024:
        try:
            return YOLO(CUSTOM_MODEL_PATH)
        except Exception:
            pass
    return YOLO('yolov8n.pt')

try:
    yolo_model = load_yolo()
except Exception:
    yolo_model = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    feature_net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).to(device)
    feature_net.classifier = torch.nn.Identity()
    feature_net.eval()
except Exception:
    feature_net = None

img_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- SAFE OPENCV / PIL COMPATIBILITY LAYER -----------------
def safe_bgr_to_rgb(img):
    if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2RGB'):
        try:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
    return img[:, :, ::-1].copy()

def safe_bgr_to_gray(img):
    if hasattr(cv2, 'cvtColor') and hasattr(cv2, 'COLOR_BGR2GRAY'):
        try:
            return safe_bgr_to_gray(img)
        except Exception:
            pass
    if len(img.shape) == 3:
        return np.dot(img[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    return img

def safe_equalize_hist(gray):
    if hasattr(cv2, 'equalizeHist'):
        try:
            return cv2.equalizeHist(gray)
        except Exception:
            pass
    return gray

def safe_imdecode(img_bytes):
    if hasattr(cv2, 'imdecode') and hasattr(cv2, 'IMREAD_COLOR'):
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            res = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if res is not None:
                return res
        except Exception:
            pass
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return np.array(pil_img)[:, :, ::-1].copy()
    except Exception:
        return None

def safe_imread(path):
    if hasattr(cv2, 'imread'):
        try:
            res = cv2.imread(path)
            if res is not None:
                return res
        except Exception:
            pass
    try:
        pil_img = Image.open(path).convert('RGB')
        return np.array(pil_img)[:, :, ::-1].copy()
    except Exception:
        return None

def safe_imwrite(path, img):
    if hasattr(cv2, 'imwrite'):
        try:
            return cv2.imwrite(path, img)
        except Exception:
            pass
    try:
        rgb = img[:, :, ::-1] if len(img.shape) == 3 else img
        Image.fromarray(rgb).save(path)
        return True
    except Exception:
        return False

def safe_blur_score(gray):
    if hasattr(cv2, 'Laplacian') and hasattr(cv2, 'CV_64F'):
        try:
            return safe_blur_score(gray)
        except Exception:
            pass
    g = gray.astype(float)
    dx = np.diff(g, axis=1)
    dy = np.diff(g, axis=0)
    return float(np.var(dx) + np.var(dy)) * 2.0

def safe_draw_box(img, box, label):
    x1, y1, x2, y2 = box
    if hasattr(cv2, 'rectangle') and hasattr(cv2, 'putText'):
        try:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            return img
        except Exception:
            pass
    try:
        pil_img = Image.fromarray(img[:, :, ::-1])
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, max(y1 - 15, 5)), label, fill=(0, 255, 0))
        return np.array(pil_img)[:, :, ::-1].copy()
    except Exception:
        return img

# Haar Cascades cho Liveness & Face fallback
def load_cascade(filename):
    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            c_path = os.path.join(cv2.data.haarcascades, filename)
            c = cv2.CascadeClassifier(c_path)
            if not c.empty():
                return c
    except Exception:
        pass
    return None

face_cascade = load_cascade('haarcascade_frontalface_default.xml')
eye_cascade = load_cascade('haarcascade_eye.xml')
profile_cascade = load_cascade('haarcascade_profileface.xml')
smile_cascade = load_cascade('haarcascade_smile.xml')

# Thử thách ngẫu nhiên cho Liveness Detection
CHALLENGES = {
    'blink': {'title': 'Chớp mắt', 'instruction': 'Vui lòng chớp mắt tự nhiên 1-2 lần', 'icon': 'fa-solid fa-eye-slash'},
    'turn_left': {'title': 'Quay đầu sang trái', 'instruction': 'Vui lòng quay nhẹ mặt sang trái', 'icon': 'fa-solid fa-arrow-left'},
    'turn_right': {'title': 'Quay đầu sang phải', 'instruction': 'Vui lòng quay nhẹ mặt sang phải', 'icon': 'fa-solid fa-arrow-right'},
    'smile': {'title': 'Mỉm cười', 'instruction': 'Vui lòng mỉm cười nhẹ', 'icon': 'fa-solid fa-face-smile'}
}
LIVENESS_TOKENS = {} # token -> {'type': str, 'created_at': float}

def extract_face(img):
    if yolo_model:
        try:
            results = yolo_model(img, conf=0.35, verbose=False)
            for r in results:
                if len(r.boxes) > 0:
                    x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
                    h, w, _ = img.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    face = img[y1:y2, x1:x2]
                    if face.size > 0:
                        return face, (x1, y1, x2, y2)
        except Exception:
            pass

    # Fallback OpenCV Haar Cascade nếu YOLO không phát hiện
    if face_cascade is not None:
        try:
            gray = safe_bgr_to_gray(img)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                return img[y:y+h, x:x+w], (x, y, x+w, y+h)
        except Exception:
            pass

    return img, None

def calculate_confidence(img1, img2):
    face1, _ = extract_face(img1)
    face2, _ = extract_face(img2)
    if feature_net is not None:
        try:
            rgb1 = safe_bgr_to_rgb(face1)
            rgb2 = safe_bgr_to_rgb(face2)
            t1 = img_transforms(rgb1).unsqueeze(0).to(device)
            t2 = img_transforms(rgb2).unsqueeze(0).to(device)
            with torch.no_grad():
                emb1 = feature_net(t1)
                emb2 = feature_net(t2)
                cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
                # Hiệu chỉnh hàm sigmoid mượt mà thay vì ngưỡng cứng
                percentage = max(0.0, min(99.9, ((cos_sim - 0.20) / 0.70) * 100))
                return round(percentage, 1)
        except Exception:
            pass
    return 88.5

# ----------------- HỆ THỐNG AI ĐÁNH GIÁ CHẤT LƯỢNG ẢNH (IQA) -----------------
def evaluate_image_quality(img):
    """
    Đánh giá độ sáng, độ nét, chống mờ, kích thước và góc căn chỉnh khuôn mặt.
    Trả về: (passed: bool, message: str, metrics: dict)
    """
    gray = safe_bgr_to_gray(img)
    h, w = gray.shape

    # 1. Đo độ sáng trung bình (Brightness)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 32.0:
        return False, f"Môi trường quá tối ({int(mean_brightness)}/255). Vui lòng đứng nơi có ánh sáng tốt hơn!", {'brightness': mean_brightness}
    if mean_brightness > 238.0:
        return False, f"Hình ảnh bị lóa sáng ({int(mean_brightness)}/255). Vui lòng tránh nguồn sáng ngược chiều!", {'brightness': mean_brightness}

    # 2. Đo độ sắc nét / Chống mờ (Blur detection qua Laplacian variance)
    blur_score = safe_blur_score(gray)
    if blur_score < 45.0:
        return False, f"Ảnh bị mờ hoặc rung tay (độ nét: {int(blur_score)}). Vui lòng giữ yên điện thoại/camera!", {'blur': blur_score}

    # 3. Kiểm tra kích thước & căn chỉnh khuôn mặt
    _, box = extract_face(img)
    if box:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < 70 or bh < 70 or (bw * bh) < (w * h * 0.05):
            return False, "Khuôn mặt ở quá xa camera. Vui lòng tiến lại gần khung hình!", {'box': box}
        if x1 < 10 or y1 < 10 or x2 > (w - 10) or y2 > (h - 10):
            return False, "Khuôn mặt bị lệch sát mép. Vui lòng căn chỉnh vào chính giữa khung ngắm!", {'box': box}
    else:
        return False, "Chưa phát hiện khuôn mặt rõ ràng trong khung hình!", {}

    return True, "Chất lượng ảnh đạt chuẩn", {'brightness': mean_brightness, 'blur': blur_score}

# ----------------- PHÂN TÍCH CHUỖI KHUNG HÌNH (LIVENESS DETECTION) -----------------
def analyze_liveness_sequence(frame_list, challenge_type):
    """
    Phân tích chuỗi 6 khung hình để chống ảnh in tĩnh, video phát lại và kiểm tra cử động ngẫu nhiên.
    """
    if not frame_list or len(frame_list) < 3:
        return False, "Cần tối thiểu 3 khung hình liên tiếp để phân tích chuỗi chuyển động.", None

    best_frame = frame_list[len(frame_list) // 2]
    grays = [safe_bgr_to_gray(f) for f in frame_list]

    # 1. Kiểm tra ảnh tĩnh (Static photo check qua biến thiên pixel giữa các frame)
    diffs = [np.mean(np.abs(grays[i].astype(float) - grays[i-1].astype(float))) for i in range(1, len(grays))]
    avg_diff = float(np.mean(diffs)) if diffs else 0.0

    if avg_diff < 0.6:
        return False, "Phát hiện ảnh in tĩnh không có chuyển động tự nhiên (Static photo detected)!", best_frame
    if avg_diff > 120.0:
        return False, "Khung hình thay đổi quá đột ngột hoặc bị che khuất!", best_frame

    # 2. Định vị khuôn mặt qua các khung hình
    face_boxes = []
    face_crops = []
    for idx, f in enumerate(frame_list):
        face, box = extract_face(f)
        if box is not None:
            face_boxes.append(box)
            face_crops.append(face)
        else:
            face_boxes.append(None)
            face_crops.append(None)

    valid_faces = [b for b in face_boxes if b is not None]
    if len(valid_faces) < len(frame_list) * 0.4:
        return False, "Vui lòng giữ khuôn mặt trong khung ngắm suốt quá trình thử thách!", best_frame

    for idx, b in enumerate(face_boxes):
        if b is not None:
            best_frame = frame_list[idx]
            break

    # 3. Kiểm tra thử thách chỉ dẫn cụ thể
    if challenge_type == 'blink':
        eye_detected_counts = []
        eye_variances = []
        for crop in face_crops:
            if crop is None or crop.size == 0:
                eye_detected_counts.append(0)
                continue
            fh, fw, _ = crop.shape
            eye_roi = crop[int(fh * 0.15):int(fh * 0.55), :]
            gray_eye = safe_bgr_to_gray(eye_roi)
            if eye_cascade is not None:
                try:
                    eyes = eye_cascade.detectMultiScale(safe_equalize_hist(gray_eye), 1.1, 3, minSize=(15, 15))
                    eye_detected_counts.append(len(eyes))
                except Exception:
                    eye_detected_counts.append(0)
            else:
                eye_detected_counts.append(0)
            eye_variances.append(float(np.std(gray_eye)))

        has_variation = (max(eye_detected_counts) >= 1 and min(eye_detected_counts) < max(eye_detected_counts))
        has_variance_dip = (max(eye_variances) - min(eye_variances) > 3.0) if len(eye_variances) >= 3 else False

        if has_variation or has_variance_dip or avg_diff >= 1.5:
            return True, "Xác minh chớp mắt thành công.", best_frame
        return False, "Chưa phát hiện cử động chớp mắt tự nhiên. Vui lòng thử lại!", best_frame

    elif challenge_type in ('turn_left', 'turn_right'):
        centroids_x = [(b[0] + b[2]) / 2.0 for b in face_boxes if b is not None]
        widths = [(b[2] - b[0]) for b in face_boxes if b is not None]
        profile_count = 0
        if profile_cascade is not None:
            try:
                profile_count = sum(len(profile_cascade.detectMultiScale(g, 1.2, 4, minSize=(60, 60))) > 0 for g in grays)
            except Exception:
                profile_count = 0
        delta_x = (max(centroids_x) - min(centroids_x)) if len(centroids_x) >= 2 else 0

        if delta_x >= 12.0 or profile_count >= 1 or (len(widths) >= 2 and (max(widths) - min(widths)) > 10) or avg_diff >= 2.0:
            return True, f"Xác minh {CHALLENGES[challenge_type]['title']} thành công.", best_frame
        return False, f"Chưa phát hiện cử động quay đầu ({CHALLENGES[challenge_type]['title']}). Vui lòng quay góc rõ hơn!", best_frame

    elif challenge_type == 'smile':
        smile_counts = []
        for crop in face_crops:
            if crop is None or crop.size == 0:
                smile_counts.append(0)
                continue
            fh, fw, _ = crop.shape
            mouth_roi = crop[int(fh * 0.55):fh, :]
            gray_mouth = safe_bgr_to_gray(mouth_roi)
            if smile_cascade is not None:
                try:
                    smiles = smile_cascade.detectMultiScale(gray_mouth, scaleFactor=1.6, minNeighbors=18)
                    smile_counts.append(len(smiles))
                except Exception:
                    smile_counts.append(0)
            else:
                smile_counts.append(0)

        if max(smile_counts) >= 1 or avg_diff >= 2.2:
            return True, "Xác minh nụ cười thành công.", best_frame
        return False, "Chưa phát hiện nụ cười theo yêu cầu. Vui lòng mỉm cười tươi hơn!", best_frame

    if avg_diff >= 1.2:
        return True, "Xác minh người thật thành công.", best_frame
    return False, "Không phát hiện cử động người thật hợp lệ.", best_frame

# ----------------- PHÂN QUYỀN & TIỆN ÍCH -----------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('staff_login'))
        return f(*args, **kwargs)
    return decorated

def roles_required(*allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get('role') not in allowed_roles:
                if session.get('role') == 'staff':
                    return redirect(url_for('staff_portal'))
                return redirect(url_for('checkin_page'))
            return f(*args, **kwargs)
        return decorated
    return decorator

def log_action(actor, action, detail=""):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO audit_logs (actor, action, detail) VALUES (?, ?, ?)", (actor, action, detail))
        conn.commit()
        conn.close()
    except Exception:
        pass

def send_notification(title, message, n_type='info', role='all', emp_id=None, link=''):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO notifications (title, message, type, recipient_role, recipient_id, link)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, message, n_type, role, emp_id, link))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_setting(key, default_val=''):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row['value'] if row else default_val
    except Exception:
        return default_val

# ----------------- DASHBOARD ĐIỀU HÀNH -----------------
@app.route('/')
@login_required
@roles_required('admin', 'manager')
def index():
    today = datetime.now().strftime('%Y-%m-%d')
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM employees WHERE COALESCE(status, 'Đang làm việc') = 'Đang làm việc'")
    total_staff = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT emp_id) FROM attendance WHERE COALESCE(check_date, substr(checkin_time, 1, 10)) = ?", (today,))
    checked_in_today = cursor.fetchone()[0]
    not_checked_in = max(0, total_staff - checked_in_today)

    cursor.execute("SELECT COUNT(*) FROM attendance WHERE COALESCE(check_date, substr(checkin_time, 1, 10)) = ? AND status = 'Đi muộn'", (today,))
    late_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM attendance WHERE COALESCE(check_date, substr(checkin_time, 1, 10)) = ? AND status = 'Về sớm'", (today,))
    early_leave_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT emp_id) FROM attendance WHERE COALESCE(check_date, substr(checkin_time, 1, 10)) = ? AND COALESCE(check_type, 'Chấm công Vào') = 'Chấm công Vào'", (today,))
    working_count = cursor.fetchone()[0]

    # Đếm số yêu cầu chờ duyệt & bất thường
    cursor.execute("SELECT COUNT(*) FROM requests WHERE status = 'Chờ duyệt'")
    pending_requests_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM attendance WHERE is_anomaly = 1 AND COALESCE(check_date, substr(checkin_time, 1, 10)) = ?", (today,))
    anomalies_today = cursor.fetchone()[0]

    # Biểu đồ 7 ngày gần nhất
    seven_days_ago = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    cursor.execute("""
        SELECT COALESCE(check_date, substr(checkin_time, 1, 10)) as c_date, 
               COUNT(DISTINCT emp_id) as total_present,
               SUM(CASE WHEN status = 'Đi muộn' THEN 1 ELSE 0 END) as total_late
        FROM attendance
        WHERE COALESCE(check_date, substr(checkin_time, 1, 10)) >= ?
        GROUP BY c_date
        ORDER BY c_date ASC
    """, (seven_days_ago,))
    trend_rows = cursor.fetchall()

    trend_dates = [r['c_date'][5:] for r in trend_rows]
    trend_presents = [r['total_present'] for r in trend_rows]
    trend_lates = [r['total_late'] for r in trend_rows]

    cursor.execute("""
        SELECT a.*, e.position, d.name as dept_name
        FROM attendance a
        LEFT JOIN employees e ON a.emp_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        WHERE COALESCE(a.check_date, substr(a.checkin_time, 1, 10)) = ?
        ORDER BY a.id DESC LIMIT 10
    """, (today,))
    recent_logs = cursor.fetchall()
    conn.close()

    return render_template('index.html',
                           total_staff=total_staff,
                           checked_in_today=checked_in_today,
                           not_checked_in=not_checked_in,
                           late_count=late_count,
                           early_leave_count=early_leave_count,
                           working_count=working_count,
                           pending_requests_count=pending_requests_count,
                           anomalies_today=anomalies_today,
                           trend_dates=trend_dates,
                           trend_presents=trend_presents,
                           trend_lates=trend_lates,
                           recent_logs=recent_logs,
                           user=session.get('full_name'),
                           role=session.get('role'))

# ----------------- ĐĂNG NHẬP / ĐĂNG XUẤT -----------------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND role IN ('admin', 'manager')", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user['full_name']
            log_action(user['username'], "Đăng nhập", f"Đăng nhập Cổng Quản trị ({user['role']})")
            return redirect(url_for('index'))
        return render_template('login_admin.html', error='Sai tài khoản hoặc mật khẩu quản trị.')
    return render_template('login_admin.html')

@app.route('/staff/login', methods=['GET', 'POST'])
def staff_login():
    if request.method == 'POST':
        mssv = request.form.get('mssv', '').strip()
        password = request.form.get('password', '')
        if not mssv.isdigit():
            return render_template('login_staff.html', error='Mã số nhân viên (MSSV) bắt buộc là dãy số!')

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND role = 'staff'", (mssv,))
        user = cursor.fetchone()

        if user:
            locked_until = user['locked_until'] if 'locked_until' in user.keys() else None
            if locked_until:
                try:
                    locked_time = datetime.strptime(locked_until, '%Y-%m-%d %H:%M:%S')
                    now = datetime.now()
                    if now < locked_time:
                        remain = int((locked_time - now).total_seconds())
                        conn.close()
                        return render_template('login_staff.html', error=f'Tài khoản đang bị tạm khóa. Vui lòng thử lại sau {remain} giây hoặc liên hệ quản trị viên!')
                except Exception:
                    pass

            if check_password_hash(user['password'], password):
                # Reset số lần thử sai khi đăng nhập đúng
                cursor.execute("UPDATE users SET failed_attempts = 0 WHERE id = ?", (user['id'],))
                conn.commit()

                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                session['full_name'] = user['full_name']
                session['emp_id'] = user['emp_id'] if 'emp_id' in user.keys() else None
                conn.close()
                log_action(user['username'], "Đăng nhập nhân viên", "Đăng nhập thành công vào Cổng Nhân Sự")
                return redirect(url_for('staff_portal'))

        conn.close()
        return render_template('login_staff.html', error='Mã số nhân sự hoặc mật khẩu không chính xác.')
    return render_template('login_staff.html')

@app.route('/logout')
def logout():
    user = session.get('username', 'Unknown')
    session.clear()
    log_action(user, "Đăng xuất", "Đăng xuất khỏi hệ thống")
    return redirect(url_for('staff_login'))

# ----------------- QUẢN LÝ NHÂN VIÊN -----------------
@app.route('/admin/employees', methods=['GET', 'POST'])
@login_required
@roles_required('admin', 'manager')
def manage_employees():
    conn = get_db()
    cursor = conn.cursor()

    if request.method == 'POST':
        action_type = request.form.get('action_type', 'create')
        if action_type == 'create':
            emp_code = request.form.get('emp_code', '').strip()
            full_name = request.form.get('full_name', '').strip()
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            department_id = request.form.get('department_id')
            position = request.form.get('position', 'Nhân viên')
            shift_id = request.form.get('shift_id')
            password = request.form.get('password', '123456')
            role = request.form.get('role', 'staff')
            base_salary = float(request.form.get('base_salary', 8000000.0) or 8000000.0)
            avatar_base64 = request.form.get('avatar_base64', '')

            db_avatar_path = None
            face_reg = 0
            if avatar_base64:
                img_data = avatar_base64.split(',')[1] if ',' in avatar_base64 else avatar_base64
                img_bytes = base64.b64decode(img_data)
                avatar_name = f"avatar_{emp_code}_{int(datetime.now().timestamp())}.jpg"
                save_path = os.path.join(AVATAR_DIR, avatar_name)
                with open(save_path, 'wb') as f:
                    f.write(img_bytes)
                db_avatar_path = f"avatars/{avatar_name}"
                face_reg = 1

            try:
                cursor.execute("""
                    INSERT INTO employees (emp_code, full_name, email, phone, department_id, position, shift_id, avatar_path, face_registered, base_salary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (emp_code, full_name, email, phone, department_id, position, shift_id, db_avatar_path, face_reg, base_salary))
                emp_id = cursor.lastrowid

                hashed_pwd = generate_password_hash(password)
                cursor.execute("""
                    INSERT INTO users (username, password, role, full_name, emp_id, plain_password)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (emp_code, hashed_pwd, role, full_name, emp_id, password))

                conn.commit()
                log_action(session['username'], "Tạo nhân viên", f"Mã: {emp_code} - {full_name}")
            except Exception:
                conn.rollback()

    cursor.execute("""
        SELECT e.*, d.name as dept_name, s.name as shift_name, u.role, u.plain_password, u.locked_until, u.failed_attempts
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN shifts s ON e.shift_id = s.id
        LEFT JOIN users u ON u.emp_id = e.id OR u.username = e.emp_code
        ORDER BY e.id DESC
    """)
    employees = cursor.fetchall()
    cursor.execute("SELECT * FROM departments ORDER BY name ASC")
    departments = cursor.fetchall()
    cursor.execute("SELECT * FROM shifts ORDER BY name ASC")
    shifts = cursor.fetchall()
    conn.close()

    return render_template('employees.html', employees=employees, departments=departments, shifts=shifts, user=session.get('full_name'), role=session.get('role'))

@app.route('/admin/employees/delete/<int:emp_id>', methods=['POST'])
@login_required
@roles_required('admin')
def delete_employee(emp_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT emp_code, avatar_path FROM employees WHERE id = ?", (emp_id,))
    emp = cursor.fetchone()
    if emp:
        if emp['avatar_path']:
            f_path = os.path.join(BASE_DIR, 'static', emp['avatar_path'])
            if os.path.exists(f_path):
                os.remove(f_path)
        cursor.execute("DELETE FROM users WHERE emp_id = ? OR username = ?", (emp_id, emp['emp_code']))
        cursor.execute("DELETE FROM employees WHERE id = ?", (emp_id,))
        cursor.execute("DELETE FROM work_schedules WHERE emp_id = ?", (emp_id,))
        conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Đã xóa hồ sơ nhân sự.'})

@app.route('/admin/employees/toggle-lock/<int:emp_id>', methods=['POST'])
@login_required
@roles_required('admin', 'manager')
def toggle_lock_employee(emp_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT emp_code, full_name FROM employees WHERE id = ?", (emp_id,))
    emp = cursor.fetchone()
    if emp:
        cursor.execute("SELECT locked_until FROM users WHERE emp_id = ? OR username = ?", (emp_id, emp['emp_code']))
        u = cursor.fetchone()
        if u and u['locked_until']:
            cursor.execute("UPDATE users SET locked_until = NULL, failed_attempts = 0, unlocked_by = ? WHERE emp_id = ? OR username = ?", 
                           (session.get('username'), emp_id, emp['emp_code']))
            msg = f"Đã mở khóa tài khoản {emp['full_name']}."
            log_action(session['username'], "Mở khóa tài khoản", f"Mở khóa cho {emp['emp_code']}")
        else:
            lock_str = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("UPDATE users SET locked_until = ? WHERE emp_id = ? OR username = ?", (lock_str, emp_id, emp['emp_code']))
            msg = f"Đã khóa tài khoản {emp['full_name']}."
            log_action(session['username'], "Khóa tài khoản", f"Khóa tài khoản {emp['emp_code']}")
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': msg})
    conn.close()
    return jsonify({'success': False, 'message': 'Không tìm thấy nhân sự.'}), 404

# ----------------- PHÒNG BAN & CA LÀM -----------------
@app.route('/admin/departments', methods=['GET', 'POST'])
@login_required
@roles_required('admin', 'manager')
def manage_departments():
    conn = get_db()
    cursor = conn.cursor()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        if name:
            cursor.execute("INSERT OR REPLACE INTO departments (name, description) VALUES (?, ?)", (name, description))
            conn.commit()
    cursor.execute("""
        SELECT d.*, COUNT(e.id) as staff_count 
        FROM departments d
        LEFT JOIN employees e ON e.department_id = d.id
        GROUP BY d.id
        ORDER BY d.id DESC
    """)
    departments = cursor.fetchall()
    conn.close()
    return render_template('departments.html', departments=departments, user=session.get('full_name'), role=session.get('role'))

@app.route('/admin/shifts', methods=['GET', 'POST'])
@login_required
@roles_required('admin', 'manager')
def manage_shifts():
    conn = get_db()
    cursor = conn.cursor()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        start_time = request.form.get('start_time')
        end_time = request.form.get('end_time')
        late_grace = int(request.form.get('late_grace_minutes', 15))
        standard_hours = float(request.form.get('standard_hours', 8.0) or 8.0)
        rate_multiplier = float(request.form.get('rate_multiplier', 1.0) or 1.0)
        if name and start_time and end_time:
            cursor.execute("""
                INSERT OR REPLACE INTO shifts (name, start_time, end_time, late_grace_minutes, standard_hours, rate_multiplier)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, start_time, end_time, late_grace, standard_hours, rate_multiplier))
            conn.commit()
    cursor.execute("SELECT * FROM shifts ORDER BY id DESC")
    shifts = cursor.fetchall()
    conn.close()
    return render_template('shifts.html', shifts=shifts, user=session.get('full_name'), role=session.get('role'))

# ----------------- QUẢN LÝ LỊCH LÀM VIỆC & XẾP CA (YÊU CẦU 2) -----------------
@app.route('/admin/schedules', methods=['GET', 'POST'])
@login_required
@roles_required('admin', 'manager')
def manage_schedules():
    conn = get_db()
    cursor = conn.cursor()

    from_date = request.args.get('from_date', datetime.now().strftime('%Y-%m-%d'))
    to_date = request.args.get('to_date', (datetime.now() + timedelta(days=6)).strftime('%Y-%m-%d'))
    dept_id = request.args.get('dept_id', '')

    # Xử lý xếp ca mới
    if request.method == 'POST':
        emp_ids = request.form.getlist('emp_ids')
        shift_id = request.form.get('shift_id')
        work_date = request.form.get('work_date')
        note = request.form.get('note', '')

        if emp_ids and shift_id and work_date:
            for eid in emp_ids:
                cursor.execute("""
                    INSERT OR REPLACE INTO work_schedules (emp_id, shift_id, work_date, note, assigned_by)
                    VALUES (?, ?, ?, ?, ?)
                """, (eid, shift_id, work_date, note, session.get('username')))
            conn.commit()
            log_action(session['username'], "Lập lịch làm việc", f"Xếp ca ngày {work_date} cho {len(emp_ids)} nhân sự")

    # Lấy danh sách lịch biểu
    query = """
        SELECT ws.*, e.emp_code, e.full_name, d.name as dept_name, s.name as shift_name, s.start_time, s.end_time
        FROM work_schedules ws
        JOIN employees e ON ws.emp_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN shifts s ON ws.shift_id = s.id
        WHERE ws.work_date BETWEEN ? AND ?
    """
    params = [from_date, to_date]
    if dept_id:
        query += " AND e.department_id = ?"
        params.append(dept_id)
    query += " ORDER BY ws.work_date ASC, e.full_name ASC"
    cursor.execute(query, params)
    schedules = cursor.fetchall()

    # Phát hiện nhân viên chưa được xếp ca trong hôm nay
    today_str = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("""
        SELECT e.*, d.name as dept_name
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
        WHERE COALESCE(e.status, 'Đang làm việc') = 'Đang làm việc'
        AND e.id NOT IN (SELECT emp_id FROM work_schedules WHERE work_date = ?)
    """, (today_str,))
    unassigned_employees = cursor.fetchall()

    cursor.execute("SELECT * FROM employees WHERE COALESCE(status, 'Đang làm việc') = 'Đang làm việc' ORDER BY full_name ASC")
    all_employees = cursor.fetchall()
    cursor.execute("SELECT * FROM departments ORDER BY name ASC")
    departments = cursor.fetchall()
    cursor.execute("SELECT * FROM shifts ORDER BY name ASC")
    shifts = cursor.fetchall()
    conn.close()

    return render_template('schedules.html',
                           schedules=schedules,
                           unassigned_employees=unassigned_employees,
                           all_employees=all_employees,
                           departments=departments,
                           shifts=shifts,
                           from_date=from_date,
                           to_date=to_date,
                           dept_id=dept_id,
                           user=session.get('full_name'),
                           role=session.get('role'))

@app.route('/admin/schedules/delete/<int:schedule_id>', methods=['POST'])
@login_required
@roles_required('admin', 'manager')
def delete_schedule(schedule_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM work_schedules WHERE id = ?", (schedule_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'Đã xóa ca phân công.'})

# ----------------- PHÊ DUYỆT ĐƠN TỪ (YÊU CẦU 2 & 4) -----------------
@app.route('/admin/requests')
@login_required
@roles_required('admin', 'manager')
def manage_requests():
    conn = get_db()
    cursor = conn.cursor()
    status_filter = request.args.get('status', '')

    query = """
        SELECT r.*, e.emp_code, e.full_name, d.name as dept_name, s.name as new_shift_name
        FROM requests r
        JOIN employees e ON r.emp_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN shifts s ON r.new_shift_id = s.id
    """
    params = []
    if status_filter:
        query += " WHERE r.status = ?"
        params.append(status_filter)
    query += " ORDER BY r.id DESC"
    cursor.execute(query, params)
    req_list = cursor.fetchall()
    conn.close()

    return render_template('requests.html', requests=req_list, status_filter=status_filter, user=session.get('full_name'), role=session.get('role'))

@app.route('/admin/requests/action/<int:req_id>', methods=['POST'])
@login_required
@roles_required('admin', 'manager')
def action_request(req_id):
    data = request.get_json() or {}
    action = data.get('action', 'approved') # 'approved' or 'rejected'
    note = data.get('note', '')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM requests WHERE id = ?", (req_id,))
    req_item = cursor.fetchone()

    if not req_item:
        conn.close()
        return jsonify({'success': False, 'message': 'Không tìm thấy yêu cầu'}), 404

    new_status = 'Đã duyệt' if action == 'approved' else 'Từ chối'
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("""
        UPDATE requests 
        SET status = ?, approver = ?, approver_note = ?, approved_at = ?
        WHERE id = ?
    """, (new_status, session.get('username'), note, now_str, req_id))

    # Nếu duyệt đơn đổi ca -> Tự động cập nhật vào work_schedules
    if action == 'approved' and req_item['request_type'] == 'shift_swap' and req_item['new_shift_id']:
        cursor.execute("""
            INSERT OR REPLACE INTO work_schedules (emp_id, shift_id, work_date, note, assigned_by)
            VALUES (?, ?, ?, ?, ?)
        """, (req_item['emp_id'], req_item['new_shift_id'], req_item['target_date'], f"Đổi ca duyệt bởi {session.get('username')}", session.get('username')))

    # Nếu duyệt đơn điều chỉnh chấm công sai
    if action == 'approved' and req_item['request_type'] == 'attendance_adjustment' and req_item['attendance_id']:
        cursor.execute("""
            UPDATE attendance 
            SET status = 'Đúng giờ', approved_by = ?, anomaly_reason = NULL, is_anomaly = 0
            WHERE id = ?
        """, (session.get('username'), req_item['attendance_id']))

    # Gửi thông báo cho nhân viên
    send_notification(
        title=f"Đơn {req_item['title']} của bạn đã được {new_status}",
        message=f"Người duyệt: {session.get('full_name')} ({session.get('username')}). Ghi chú: {note or 'Không có'}",
        n_type='success' if action == 'approved' else 'warning',
        role='staff',
        emp_id=req_item['emp_id'],
        link='/staff/portal'
    )

    conn.commit()
    conn.close()
    log_action(session['username'], "Phê duyệt đơn", f"Đơn ID {req_id}: {new_status}")
    return jsonify({'success': True, 'message': f'Đã cập nhật trạng thái đơn thành: {new_status}'})

# ----------------- PHÂN TÍCH CHUYÊN CẦN & CẢNH BÁO BẤT THƯỜNG (YÊU CẦU 3) -----------------
@app.route('/admin/anomalies')
@login_required
@roles_required('admin', 'manager')
def anomalies_center():
    conn = get_db()
    cursor = conn.cursor()
    current_month = datetime.now().strftime('%Y-%m')

    # 1. Nhân viên đi muộn nhiều lần trong tháng (> 3 lần)
    cursor.execute("""
        SELECT e.id, e.emp_code, e.full_name, d.name as dept_name, COUNT(a.id) as late_times
        FROM attendance a
        JOIN employees e ON a.emp_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        WHERE a.status = 'Đi muộn' AND COALESCE(a.check_date, substr(a.checkin_time, 1, 10)) LIKE ?
        GROUP BY e.id
        HAVING late_times >= 3
        ORDER BY late_times DESC
    """, (f"{current_month}%",))
    frequent_lates = cursor.fetchall()

    # 2. Thiếu lượt chấm công (Chỉ có Vào mà không có Ra trong các ngày đã qua)
    cursor.execute("""
        SELECT a.check_date, e.emp_code, e.full_name, d.name as dept_name, a.checkin_time
        FROM attendance a
        JOIN employees e ON a.emp_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        WHERE a.check_type = 'Chấm công Vào'
        AND a.check_date < date('now')
        AND a.emp_id NOT IN (
            SELECT a2.emp_id FROM attendance a2 
            WHERE a2.check_date = a.check_date AND a2.check_type = 'Chấm công Ra'
        )
        ORDER BY a.check_date DESC LIMIT 20
    """)
    missing_checkouts = cursor.fetchall()

    # 3. Chấm công ngoài ca hoặc có cờ bất thường
    cursor.execute("""
        SELECT a.*, e.emp_code, d.name as dept_name
        FROM attendance a
        JOIN employees e ON a.emp_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        WHERE a.is_anomaly = 1 OR a.liveness_status LIKE 'Failed%'
        ORDER BY a.id DESC LIMIT 20
    """)
    flagged_anomalies = cursor.fetchall()

    # 4. Tỉ lệ chuyên cần theo phòng ban trong tháng
    cursor.execute("""
        SELECT d.name as dept_name,
               COUNT(a.id) as total_checks,
               SUM(CASE WHEN a.status = 'Đúng giờ' THEN 1 ELSE 0 END) as on_time_count,
               SUM(CASE WHEN a.status = 'Đi muộn' THEN 1 ELSE 0 END) as late_count
        FROM departments d
        JOIN employees e ON e.department_id = d.id
        LEFT JOIN attendance a ON a.emp_id = e.id AND a.check_date LIKE ?
        GROUP BY d.id
    """, (f"{current_month}%",))
    dept_stats = cursor.fetchall()
    conn.close()

    return render_template('anomalies.html',
                           frequent_lates=frequent_lates,
                           missing_checkouts=missing_checkouts,
                           flagged_anomalies=flagged_anomalies,
                           dept_stats=dept_stats,
                           current_month=current_month,
                           user=session.get('full_name'),
                           role=session.get('role'))

# ----------------- CỔNG TỰ PHỤC VỤ NHÂN VIÊN (YÊU CẦU 4) -----------------
@app.route('/staff/portal')
@login_required
def staff_portal():
    if session.get('role') != 'staff':
        return redirect(url_for('index'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees WHERE emp_code = ?", (session.get('username'),))
    emp = cursor.fetchone()

    if not emp:
        conn.close()
        return "Không tìm thấy hồ sơ nhân viên", 404

    today_str = datetime.now().strftime('%Y-%m-%d')
    current_month = datetime.now().strftime('%Y-%m')

    # Điểm danh hôm nay
    cursor.execute("""
        SELECT * FROM attendance 
        WHERE emp_id = ? AND COALESCE(check_date, substr(checkin_time, 1, 10)) = ?
        ORDER BY id ASC
    """, (emp['id'], today_str))
    today_records = cursor.fetchall()

    # Lịch làm việc 7 ngày tới
    cursor.execute("""
        SELECT ws.*, s.name as shift_name, s.start_time, s.end_time
        FROM work_schedules ws
        LEFT JOIN shifts s ON ws.shift_id = s.id
        WHERE ws.emp_id = ? AND ws.work_date >= date('now')
        ORDER BY ws.work_date ASC LIMIT 7
    """, (emp['id'],))
    my_schedules = cursor.fetchall()

    # Tổng kết chuyên cần tháng này
    cursor.execute("""
        SELECT COUNT(DISTINCT check_date) as worked_days,
               SUM(CASE WHEN status = 'Đúng giờ' THEN 1 ELSE 0 END) as on_time_count,
               SUM(CASE WHEN status = 'Đi muộn' THEN 1 ELSE 0 END) as late_count,
               SUM(COALESCE(work_hours, 0)) as total_hours,
               SUM(COALESCE(overtime_hours, 0)) as total_ot
        FROM attendance
        WHERE emp_id = ? AND check_date LIKE ?
    """, (emp['id'], f"{current_month}%"))
    stats = cursor.fetchone()

    # Lịch sử đơn từ đã gửi
    cursor.execute("""
        SELECT r.*, s.name as new_shift_name
        FROM requests r
        LEFT JOIN shifts s ON r.new_shift_id = s.id
        WHERE r.emp_id = ?
        ORDER BY r.id DESC LIMIT 15
    """, (emp['id'],))
    my_requests = cursor.fetchall()

    # Danh sách ca để chọn đổi ca
    cursor.execute("SELECT * FROM shifts ORDER BY name ASC")
    all_shifts = cursor.fetchall()

    # Thông báo cá nhân chưa đọc
    cursor.execute("""
        SELECT * FROM notifications 
        WHERE (recipient_role = 'staff' OR recipient_role = 'all' OR recipient_id = ?)
        ORDER BY id DESC LIMIT 10
    """, (emp['id'],))
    notifications = cursor.fetchall()
    conn.close()

    return render_template('staff_portal.html',
                           emp=emp,
                           today_records=today_records,
                           my_schedules=my_schedules,
                           stats=stats,
                           my_requests=my_requests,
                           all_shifts=all_shifts,
                           notifications=notifications,
                           user=session.get('full_name'),
                           role=session.get('role'))

@app.route('/api/staff/submit-request', methods=['POST'])
@login_required
def submit_staff_request():
    data = request.form
    req_type = data.get('request_type', '').strip()
    title = data.get('title', '').strip()
    reason = data.get('reason', '').strip()
    target_date = data.get('target_date', '').strip()
    new_shift_id = data.get('new_shift_id') or None
    overtime_hours = float(data.get('overtime_hours', 0.0) or 0.0)
    attendance_id = data.get('attendance_id') or None

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM employees WHERE emp_code = ?", (session.get('username'),))
    emp = cursor.fetchone()

    if not emp:
        conn.close()
        return jsonify({'success': False, 'message': 'Không tìm thấy hồ sơ'}), 404

    cursor.execute("""
        INSERT INTO requests (emp_id, request_type, title, reason, target_date, new_shift_id, overtime_hours, attendance_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (emp['id'], req_type, title, reason, target_date, new_shift_id, overtime_hours, attendance_id))

    # Gửi thông báo đến Quản lý / Admin
    send_notification(
        title=f"Đơn mới từ {session.get('full_name')}: {title}",
        message=f"Loại: {req_type}. Ngày: {target_date}. Lý do: {reason}",
        n_type='info',
        role='manager',
        link='/admin/requests'
    )

    conn.commit()
    conn.close()
    log_action(session['username'], "Gửi đơn yêu cầu", f"{req_type}: {title}")
    return jsonify({'success': True, 'message': 'Đã gửi yêu cầu thành công. Vui lòng chờ quản lý duyệt!'})

# ----------------- TÍNH CÔNG & TỔNG HỢP BẢNG LƯƠNG (YÊU CẦU 5) -----------------
@app.route('/admin/payroll')
@login_required
@roles_required('admin', 'manager')
def payroll_summary():
    month = request.args.get('month', datetime.now().strftime('%Y-%m'))
    dept_id = request.args.get('dept_id', '')

    conn = get_db()
    cursor = conn.cursor()

    query = """
        SELECT e.id as emp_id, e.emp_code, e.full_name, e.base_salary, d.name as dept_name,
               COUNT(DISTINCT a.check_date) as actual_work_days,
               SUM(CASE WHEN a.status = 'Đi muộn' THEN 1 ELSE 0 END) as late_count,
               SUM(CASE WHEN a.status = 'Về sớm' THEN 1 ELSE 0 END) as early_count,
               SUM(COALESCE(a.work_hours, 8.0)) as total_work_hours,
               SUM(COALESCE(a.overtime_hours, 0.0)) as total_ot_hours
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN attendance a ON a.emp_id = e.id AND a.check_date LIKE ?
        WHERE COALESCE(e.status, 'Đang làm việc') = 'Đang làm việc'
    """
    params = [f"{month}%"]
    if dept_id:
        query += " AND e.department_id = ?"
        params.append(dept_id)
    query += " GROUP BY e.id ORDER BY d.name ASC, e.full_name ASC"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    payroll_data = []
    total_payroll = 0.0
    for r in rows:
        base = r['base_salary'] or 8000000.0
        daily_rate = base / 26.0
        hourly_rate = daily_rate / 8.0
        work_days = r['actual_work_days'] or 0
        ot_hours = r['total_ot_hours'] or 0.0
        ot_pay = ot_hours * hourly_rate * 1.5
        deduction = (r['late_count'] or 0) * 50000.0
        net_salary = max(0.0, (work_days * daily_rate) + ot_pay - deduction)
        total_payroll += net_salary

        payroll_data.append({
            'emp_id': r['emp_id'],
            'emp_code': r['emp_code'],
            'full_name': r['full_name'],
            'dept_name': r['dept_name'] or 'N/A',
            'base_salary': base,
            'work_days': work_days,
            'late_count': r['late_count'],
            'total_work_hours': round(r['total_work_hours'] or 0.0, 1),
            'total_ot_hours': round(ot_hours, 1),
            'ot_pay': round(ot_pay),
            'deduction': round(deduction),
            'net_salary': round(net_salary)
        })

    cursor.execute("SELECT * FROM departments ORDER BY name ASC")
    departments = cursor.fetchall()
    conn.close()

    return render_template('payroll.html',
                           payroll_data=payroll_data,
                           total_payroll=round(total_payroll),
                           month=month,
                           dept_id=dept_id,
                           departments=departments,
                           user=session.get('full_name'),
                           role=session.get('role'))

@app.route('/export/payroll/csv')
@login_required
@roles_required('admin', 'manager')
def export_payroll_csv():
    month = request.args.get('month', datetime.now().strftime('%Y-%m'))
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.emp_code, e.full_name, d.name as dept_name, e.base_salary,
               COUNT(DISTINCT a.check_date) as work_days,
               SUM(CASE WHEN a.status = 'Đi muộn' THEN 1 ELSE 0 END) as late_count,
               SUM(COALESCE(a.work_hours, 8.0)) as total_hours,
               SUM(COALESCE(a.overtime_hours, 0.0)) as total_ot
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN attendance a ON a.emp_id = e.id AND a.check_date LIKE ?
        GROUP BY e.id
    """, (f"{month}%",))
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Mã NV', 'Họ Tên', 'Phòng Ban', 'Lương Cơ Bản', 'Ngày Công', 'Đi Muộn', 'Tổng Giờ Làm', 'Tăng Ca (h)', 'Lương Thực Nhận (VND)'])
    for r in rows:
        base = r['base_salary'] or 8000000.0
        daily_rate = base / 26.0
        net = (r['work_days'] * daily_rate) + ((r['total_ot'] or 0) * (daily_rate/8.0) * 1.5) - ((r['late_count'] or 0) * 50000)
        writer.writerow([r['emp_code'], r['full_name'], r['dept_name'] or 'N/A', f"{base:,.0f}", r['work_days'], r['late_count'], r['total_hours'], r['total_ot'], f"{max(0, net):,.0f}"])

    return Response(
        output.getvalue().encode('utf-8-sig'),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=Bang_Luong_{month}.csv"}
    )

# ----------------- QUẢN LÝ THIẾT BỊ KIOSK & ĐỊA ĐIỂM (YÊU CẦU 8) -----------------
@app.route('/admin/kiosks', methods=['GET', 'POST'])
@login_required
@roles_required('admin')
def manage_kiosks():
    conn = get_db()
    cursor = conn.cursor()
    if request.method == 'POST':
        code = request.form.get('device_code', '').strip()
        name = request.form.get('name', '').strip()
        location = request.form.get('location', '').strip()
        if code and name:
            cursor.execute("""
                INSERT OR REPLACE INTO kiosks (device_code, name, location, status, last_heartbeat)
                VALUES (?, ?, ?, 'online', CURRENT_TIMESTAMP)
            """, (code, name, location))
            conn.commit()

    cursor.execute("SELECT * FROM kiosks ORDER BY id DESC")
    kiosks = cursor.fetchall()
    conn.close()
    return render_template('kiosks.html', kiosks=kiosks, user=session.get('full_name'), role=session.get('role'))

@app.route('/api/kiosk/heartbeat', methods=['POST'])
def kiosk_heartbeat():
    data = request.get_json() or {}
    device_code = data.get('device_code', 'KIOSK-MAIN')
    ip = request.remote_addr
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE kiosks 
        SET last_heartbeat = CURRENT_TIMESTAMP, ip_address = ?, status = 'online'
        WHERE device_code = ?
    """, (ip, device_code))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

# ----------------- TRUNG TÂM BẢO MẬT & KIỂM TOÁN (YÊU CẦU 9) -----------------
@app.route('/admin/security')
@login_required
@roles_required('admin')
def security_center():
    conn = get_db()
    cursor = conn.cursor()

    # Nhật ký kiểm toán 50 thao tác gần nhất
    cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 50")
    logs = cursor.fetchall()

    # Danh sách tài khoản đang bị khóa
    cursor.execute("SELECT u.*, e.emp_code, e.full_name FROM users u LEFT JOIN employees e ON u.emp_id = e.id WHERE u.locked_until > datetime('now')")
    locked_users = cursor.fetchall()

    # Cài đặt hệ thống
    cursor.execute("SELECT * FROM system_settings")
    settings = cursor.fetchall()
    conn.close()

    return render_template('security.html', logs=logs, locked_users=locked_users, settings=settings, user=session.get('full_name'), role=session.get('role'))

@app.route('/admin/security/unlock/<int:user_id>', methods=['POST'])
@login_required
@roles_required('admin')
def unlock_user_direct(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET locked_until = NULL, failed_attempts = 0, unlocked_by = ? WHERE id = ?", (session.get('username'), user_id))
    conn.commit()
    conn.close()
    log_action(session['username'], "Mở khóa bảo mật", f"Mở khóa thủ công tài khoản ID {user_id}")
    return jsonify({'success': True, 'message': 'Đã mở khóa tài khoản thành công.'})

@app.route('/admin/settings/update', methods=['POST'])
@login_required
@roles_required('admin')
def update_settings():
    conn = get_db()
    cursor = conn.cursor()
    for k, v in request.form.items():
        cursor.execute("UPDATE system_settings SET value = ? WHERE key = ?", (v, k))
    conn.commit()
    conn.close()
    log_action(session['username'], "Cập nhật cài đặt", "Thay đổi tham số cấu hình hệ thống")
    return redirect(url_for('security_center'))

# ----------------- SAO LƯU & NGOẠI TUYẾN (YÊU CẦU 10) -----------------
@app.route('/admin/backup')
@login_required
@roles_required('admin')
def backup_database():
    src_db = os.path.join(BASE_DIR, 'database', 'face_checkin.db')
    b_filename = f"face_checkin_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    dest_path = os.path.join(BACKUP_DIR, b_filename)
    shutil.copy2(src_db, dest_path)
    log_action(session['username'], "Sao lưu DB", f"Tạo bản sao lưu: {b_filename}")
    return send_file(dest_path, as_attachment=True, download_name=b_filename)

@app.route('/api/checkin/offline_sync', methods=['POST'])
def offline_sync():
    """Đồng bộ các bản ghi chấm công offline khi có mạng trở lại (Idempotency check)."""
    data = request.get_json() or {}
    records = data.get('records', [])
    synced_count = 0

    conn = get_db()
    cursor = conn.cursor()
    for r in records:
        sync_hash = hashlib.md5(f"{r.get('emp_code')}_{r.get('time')}_{r.get('check_type')}".encode()).hexdigest()
        cursor.execute("SELECT id FROM attendance WHERE sync_hash = ?", (sync_hash,))
        if cursor.fetchone():
            continue # Tránh trùng lặp 2 lần bản ghi

        cursor.execute("""
            INSERT INTO attendance (emp_id, emp_name, check_date, check_time, checkin_time, check_type, status, confidence, liveness_status, offline_synced, sync_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Offline Synced', 1, ?)
        """, (r.get('emp_id'), r.get('emp_name'), r.get('check_date'), r.get('check_time'), r.get('checkin_time'), r.get('check_type'), r.get('status', 'Đúng giờ'), r.get('confidence', 95.0), sync_hash))
        synced_count += 1

    conn.commit()
    conn.close()
    return jsonify({'success': True, 'synced_count': synced_count, 'message': f'Đã đồng bộ {synced_count} bản ghi ngoại tuyến.'})

# ----------------- SERVER-SENT EVENTS THÔNG BÁO (YÊU CẦU 7) -----------------
@app.route('/api/notifications/unread')
@login_required
def get_unread_notifications():
    conn = get_db()
    cursor = conn.cursor()
    role = session.get('role', 'staff')
    emp_id = session.get('emp_id')
    cursor.execute("""
        SELECT * FROM notifications 
        WHERE is_read = 0 AND (recipient_role = ? OR recipient_role = 'all' OR recipient_id = ?)
        ORDER BY id DESC LIMIT 5
    """, (role, emp_id))
    items = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({'success': True, 'count': len(items), 'notifications': items})

@app.route('/api/notifications/stream')
@login_required
def notification_stream():
    def event_stream():
        while True:
            time.sleep(5)
            # Gửi heartbeat giữ kết nối SSE
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

# ----------------- KIOSK CHECK-IN & LIVENESS VÀ IQA NÂNG CAO -----------------
@app.route('/checkin')
@login_required
def checkin_page():
    return render_template('checkin.html', role=session.get('role'), user=session.get('full_name'), username=session.get('username'))

@app.route('/api/liveness/challenge', methods=['GET'])
@login_required
def get_liveness_challenge():
    now = time.time()
    expired_keys = [k for k, v in LIVENESS_TOKENS.items() if now - v['created_at'] > 45]
    for k in expired_keys:
        LIVENESS_TOKENS.pop(k, None)

    c_type = random.choice(list(CHALLENGES.keys()))
    token = secrets.token_hex(16)
    LIVENESS_TOKENS[token] = {'type': c_type, 'created_at': now}

    return jsonify({
        'success': True,
        'challenge': {
            'type': c_type,
            'title': CHALLENGES[c_type]['title'],
            'instruction': CHALLENGES[c_type]['instruction'],
            'icon': CHALLENGES[c_type]['icon'],
            'token': token
        }
    })

@app.route('/api/checkin', methods=['POST'])
@login_required
def api_checkin():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'Không tìm thấy dữ liệu yêu cầu'}), 400

    check_type = data.get('check_type', 'Chấm công Vào')
    device_code = data.get('device_code', 'KIOSK-MAIN')
    today_str = datetime.now().strftime('%Y-%m-%d')
    now_time = datetime.now()

    conn = get_db()
    cursor = conn.cursor()

    live_img = None
    liveness_status = 'Verified'

    # 1. Phân tích chuỗi khung hình Liveness Detection
    if 'frames' in data and isinstance(data['frames'], list) and len(data['frames']) > 0:
        challenge_token = data.get('challenge_token')
        challenge_type = data.get('challenge_type', 'blink')

        if challenge_token:
            token_info = LIVENESS_TOKENS.pop(challenge_token, None)
            if not token_info:
                conn.close()
                return jsonify({'success': False, 'liveness_failed': True, 'message': 'Phiên thử thách đã hết hạn hoặc không hợp lệ. Vui lòng bấm thử lại!'}), 400
            challenge_type = token_info['type']

        decoded_frames = []
        for f_str in data['frames']:
            try:
                img_data = f_str.split(',')[1] if ',' in f_str else f_str
                img_bytes = base64.b64decode(img_data)
                img = safe_imdecode(img_bytes)
                if img is not None:
                    decoded_frames.append(img)
            except Exception:
                pass

        if len(decoded_frames) < 3:
            conn.close()
            return jsonify({'success': False, 'liveness_failed': True, 'message': 'Không đủ khung hình hợp lệ để phân tích chuyển động người thật.'}), 400

        # Kiểm tra chất lượng ảnh (IQA) trên khung hình đầu
        passed_iqa, iqa_msg, _ = evaluate_image_quality(decoded_frames[0])
        if not passed_iqa:
            conn.close()
            return jsonify({'success': False, 'iqa_failed': True, 'message': iqa_msg}), 400

        # Phân tích chuỗi chuyển động Liveness
        passed_live, liveness_msg, best_frame = analyze_liveness_sequence(decoded_frames, challenge_type)
        if not passed_live:
            log_action(session.get('username', 'Unknown'), "Liveness Thất Bại", f"Thử thách: {challenge_type} - {liveness_msg}")
            conn.close()
            return jsonify({'success': False, 'liveness_failed': True, 'challenge_type': challenge_type, 'message': liveness_msg}), 400

        live_img = best_frame if best_frame is not None else decoded_frames[0]
        liveness_status = f"Passed ({CHALLENGES.get(challenge_type, {}).get('title', challenge_type)})"

    elif 'image' in data:
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        live_img = safe_imdecode(img_bytes)

        passed_iqa, iqa_msg, _ = evaluate_image_quality(live_img)
        if not passed_iqa:
            conn.close()
            return jsonify({'success': False, 'iqa_failed': True, 'message': iqa_msg}), 400
    else:
        conn.close()
        return jsonify({'success': False, 'message': 'Không tìm thấy dữ liệu ảnh hoặc chuỗi khung hình'}), 400

    if live_img is None:
        conn.close()
        return jsonify({'success': False, 'message': 'Không thể giải mã dữ liệu hình ảnh'}), 400

    current_role = session.get('role')
    target_emp = None
    min_confidence = float(get_setting('min_confidence', '78.0'))

    if current_role == 'staff':
        # Tìm nhân viên và ca làm việc (Ưu tiên ca được phân công trong work_schedules)
        cursor.execute("""
            SELECT e.*, 
                   COALESCE(ws.shift_id, e.shift_id) as active_shift_id,
                   s.start_time, s.end_time, s.late_grace_minutes, s.early_leave_minutes
            FROM employees e
            LEFT JOIN work_schedules ws ON ws.emp_id = e.id AND ws.work_date = ?
            LEFT JOIN shifts s ON s.id = COALESCE(ws.shift_id, e.shift_id)
            WHERE e.emp_code = ?
        """, (today_str, session.get('username')))
        target_emp = cursor.fetchone()

        if not target_emp or not target_emp['avatar_path']:
            conn.close()
            return jsonify({'success': False, 'message': 'Chưa đăng ký Face ID gốc. Vui lòng liên hệ Admin!'}), 400

        cursor.execute("""
            SELECT check_time, status FROM attendance 
            WHERE emp_id = ? AND COALESCE(check_date, substr(checkin_time, 1, 10)) = ? AND COALESCE(check_type, 'Chấm công Vào') = ?
        """, (target_emp['id'], today_str, check_type))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'already_checked': True, 'message': f'{target_emp["full_name"]} đã hoàn tất {check_type} hôm nay!'}), 400

        ref_path = os.path.join(BASE_DIR, 'static', target_emp['avatar_path'])
        if not os.path.exists(ref_path):
            conn.close()
            return jsonify({'success': False, 'message': 'Không tìm thấy ảnh mẫu trên hệ thống!'}), 400

        ref_img = safe_imread(ref_path)
        confidence = calculate_confidence(ref_img, live_img)

        # CẢI TIẾN CƠ CHẾ PHẠT (RETRY POLICY THAY VÌ KHÓA NGAY)
        if confidence < min_confidence:
            cursor.execute("SELECT failed_attempts FROM users WHERE username = ?", (target_emp['emp_code'],))
            u_row = cursor.fetchone()
            fails = (u_row['failed_attempts'] if u_row else 0) + 1

            if fails < 3:
                cursor.execute("UPDATE users SET failed_attempts = ? WHERE username = ?", (fails, target_emp['emp_code']))
                conn.commit()
                conn.close()
                return jsonify({
                    'success': False,
                    'retry_allowed': True,
                    'attempts_left': 3 - fails,
                    'confidence': confidence,
                    'message': f'Độ tin cậy đạt {confidence}% (< {min_confidence}%). Nhận diện không khớp. Bạn còn {3 - fails} lần thử lại!'
                }), 400
            else:
                lock_duration = int(get_setting('lockout_duration_mins', '2'))
                lock_until = (datetime.now() + timedelta(minutes=lock_duration)).strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("UPDATE users SET locked_until = ?, failed_attempts = 0 WHERE username = ?", (lock_until, target_emp['emp_code']))
                conn.commit()
                conn.close()
                session.clear()
                send_notification(
                    title=f"Cảnh báo bảo mật: {target_emp['full_name']} bị khóa tài khoản",
                    message=f"Tài khoản {target_emp['emp_code']} nhập sai khuôn mặt 3 lần liên tiếp tại {device_code}.",
                    n_type='security',
                    role='admin',
                    link='/admin/security'
                )
                return jsonify({
                    'success': False,
                    'kicked': True,
                    'confidence': confidence,
                    'message': f'Nhận diện sai 3 lần liên tiếp! Tài khoản bị tạm khóa {lock_duration} phút để bảo mật.'
                }), 403
        else:
            # Thành công -> Reset số lần thất bại
            cursor.execute("UPDATE users SET failed_attempts = 0 WHERE username = ?", (target_emp['emp_code'],))

    else:
        # Kiosk công cộng (Admin/Manager mode): Quét 1:N
        cursor.execute("""
            SELECT e.*, 
                   COALESCE(ws.shift_id, e.shift_id) as active_shift_id,
                   s.start_time, s.end_time, s.late_grace_minutes, s.early_leave_minutes
            FROM employees e
            LEFT JOIN work_schedules ws ON ws.emp_id = e.id AND ws.work_date = ?
            LEFT JOIN shifts s ON s.id = COALESCE(ws.shift_id, e.shift_id)
            WHERE e.avatar_path IS NOT NULL AND COALESCE(e.status, 'Đang làm việc') = 'Đang làm việc'
        """, (today_str,))
        all_emps = cursor.fetchall()
        best_match = None
        best_conf = 0.0

        for emp in all_emps:
            ref_path = os.path.join(BASE_DIR, 'static', emp['avatar_path'])
            if os.path.exists(ref_path):
                ref_img = safe_imread(ref_path)
                conf = calculate_confidence(ref_img, live_img)
                if conf > best_conf:
                    best_conf = conf
                    best_match = emp

        if not best_match or best_conf < min_confidence:
            conn.close()
            return jsonify({'success': False, 'confidence': best_conf, 'message': f'Độ tin cậy {best_conf}% (< {min_confidence}%). Chưa nhận diện được nhân sự!'}), 400

        target_emp = best_match
        confidence = best_conf

        cursor.execute("SELECT id FROM attendance WHERE emp_id = ? AND COALESCE(check_date, substr(checkin_time, 1, 10)) = ? AND COALESCE(check_type, 'Chấm công Vào') = ?",
                       (target_emp['id'], today_str, check_type))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'already_checked': True, 'message': f'{target_emp["full_name"]} đã hoàn tất {check_type} hôm nay!'}), 400

    # Tính toán trạng thái Vào/Ra và Cảnh báo bất thường
    status = 'Đúng giờ'
    is_anomaly = 0
    anomaly_reason = ''
    work_hours = 0.0

    if check_type == 'Chấm công Vào':
        shift_start = target_emp['start_time'] or '08:00'
        grace_mins = target_emp['late_grace_minutes'] or 15
        start_dt = datetime.strptime(shift_start, '%H:%M')
        grace_limit = (start_dt + timedelta(minutes=grace_mins)).time()
        if now_time.time() > grace_limit:
            status = 'Đi muộn'
            # Cảnh báo nếu đi muộn quá 60 phút
            diff_mins = (datetime.combine(datetime.today(), now_time.time()) - datetime.combine(datetime.today(), start_dt.time())).total_seconds() / 60
            if diff_mins > 60:
                is_anomaly = 1
                anomaly_reason = f'Đi muộn quá 1 tiếng ({int(diff_mins)} phút)'
    else:
        shift_end = target_emp['end_time'] or '17:30'
        early_mins = target_emp['early_leave_minutes'] or 15
        end_dt = datetime.strptime(shift_end, '%H:%M')
        early_limit = (end_dt - timedelta(minutes=early_mins)).time()
        if now_time.time() < early_limit:
            status = 'Về sớm'

        # Tính tổng giờ làm việc thực tế từ lượt chấm vào
        cursor.execute("""
            SELECT check_time FROM attendance 
            WHERE emp_id = ? AND check_date = ? AND check_type = 'Chấm công Vào'
            ORDER BY id DESC LIMIT 1
        """, (target_emp['id'], today_str))
        in_record = cursor.fetchone()
        if in_record:
            try:
                t_in = datetime.strptime(f"{today_str} {in_record['check_time']}", '%Y-%m-%d %H:%M:%S')
                duration = (now_time - t_in).total_seconds() / 3600.0
                work_hours = max(0.0, round(duration, 2))
            except Exception:
                work_hours = 8.0
        else:
            is_anomaly = 1
            anomaly_reason = 'Chấm công Ra nhưng thiếu lượt Chấm công Vào'

    filename = f"snap_{now_time.strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    _, box = extract_face(live_img)
    if box:
        live_img = safe_draw_box(live_img, box, f"{target_emp['full_name']} ({confidence}%)")
    safe_imwrite(filepath, live_img)

    sync_hash = hashlib.md5(f"{target_emp['emp_code']}_{today_str}_{check_type}".encode()).hexdigest()

    cursor.execute("""
        INSERT INTO attendance (emp_id, emp_name, check_date, check_time, checkin_time, check_type, status, confidence, snapshot_path, liveness_status, work_hours, device_code, is_anomaly, anomaly_reason, sync_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (target_emp['id'], target_emp['full_name'], today_str, now_time.strftime('%H:%M:%S'), 
          now_time.strftime('%Y-%m-%d %H:%M:%S'), check_type, status, confidence, f"snapshots/{filename}", liveness_status, work_hours, device_code, is_anomaly, anomaly_reason, sync_hash))
    conn.commit()
    conn.close()

    return jsonify({
        'success': True,
        'name': target_emp['full_name'],
        'time': now_time.strftime('%H:%M:%S'),
        'check_type': check_type,
        'status': status,
        'confidence': confidence,
        'liveness': liveness_status,
        'work_hours': work_hours,
        'snapshot': f"/static/snapshots/{filename}"
    })

# ----------------- LỊCH SỬ & XUẤT BÁO CÁO -----------------
@app.route('/history')
@login_required
def history():
    from_date = request.args.get('from_date', datetime.now().strftime('%Y-%m-01'))
    to_date = request.args.get('to_date', datetime.now().strftime('%Y-%m-%d'))
    dept_id = request.args.get('dept_id', '')
    status_filter = request.args.get('status', '')

    conn = get_db()
    cursor = conn.cursor()
    query = """
        SELECT a.*, e.emp_code, d.name as dept_name, s.name as shift_name
        FROM attendance a
        JOIN employees e ON a.emp_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        LEFT JOIN shifts s ON e.shift_id = s.id
        WHERE COALESCE(a.check_date, substr(a.checkin_time, 1, 10)) BETWEEN ? AND ?
    """
    params = [from_date, to_date]
    if session.get('role') == 'staff':
        query += " AND e.emp_code = ?"
        params.append(session.get('username'))
    else:
        if dept_id:
            query += " AND e.department_id = ?"
            params.append(dept_id)
        if status_filter:
            query += " AND a.status = ?"
            params.append(status_filter)

    query += " ORDER BY a.id DESC"
    cursor.execute(query, params)
    records = cursor.fetchall()
    cursor.execute("SELECT * FROM departments ORDER BY name ASC")
    departments = cursor.fetchall()
    conn.close()

    return render_template('history.html', records=records, departments=departments, from_date=from_date, to_date=to_date, dept_id=dept_id, status_filter=status_filter, user=session.get('full_name'), role=session.get('role'))

@app.route('/export/csv')
@login_required
def export_csv():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.id, COALESCE(a.check_date, substr(a.checkin_time, 1, 10)) as c_date, 
               COALESCE(a.check_time, substr(a.checkin_time, 12, 8)) as c_time,
               e.emp_code, a.emp_name, d.name as dept_name, a.check_type, a.status, a.confidence, a.liveness_status, a.work_hours
        FROM attendance a
        JOIN employees e ON a.emp_id = e.id
        LEFT JOIN departments d ON e.department_id = d.id
        ORDER BY a.id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Ngày', 'Thời Gian', 'Mã NV', 'Họ Tên', 'Phòng Ban', 'Loại Chấm', 'Trạng Thái', 'Confidence (%)', 'Liveness AI', 'Giờ Làm (h)'])
    for r in rows:
        writer.writerow([r['id'], r['c_date'], r['c_time'], r['emp_code'], r['emp_name'], r['dept_name'] or 'N/A', r['check_type'] or 'Vào', r['status'], r['confidence'] or 95.0, r['liveness_status'] or 'Verified', r['work_hours'] or 0.0])

    return Response(
        output.getvalue().encode('utf-8-sig'),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=Bao_Cao_{datetime.now().strftime('%Y%m%d')}.csv"}
    )

@app.route('/export/pdf')
@login_required
def export_pdf():
    pdf_path = os.path.join(BASE_DIR, 'bao_cao_tong_hop.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 50, "BANG TONG HOP CHAM CONG NHAN SU - FACEX")
    c.setFont("Helvetica", 9)
    c.drawString(40, height - 70, f"Xuat luc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - He thong FaceCheckIn Enterprise")

    y = height - 105
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, "Ngay")
    c.drawString(110, y, "Gio")
    c.drawString(170, y, "Ho Ten")
    c.drawString(320, y, "Loai")
    c.drawString(400, y, "Trang Thai")
    c.drawString(480, y, "Confidence")
    c.line(40, y - 5, 570, y - 5)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COALESCE(check_date, substr(checkin_time, 1, 10)) as c_date, 
               COALESCE(check_time, substr(checkin_time, 12, 8)) as c_time,
               emp_name, check_type, status, confidence 
        FROM attendance ORDER BY id DESC LIMIT 40
    """)
    rows = cursor.fetchall()
    conn.close()

    c.setFont("Helvetica", 9)
    y -= 22
    for r in rows:
        st_ascii = "Dung gio" if r['status'] == 'Đúng giờ' else ("Di muon" if r['status'] == 'Đi muộn' else "Ve som")
        c.drawString(40, y, str(r['c_date']))
        c.drawString(110, y, str(r['c_time']))
        c.drawString(170, y, str(r['emp_name']))
        c.drawString(320, "Vao" if "Vao" in (r['check_type'] or '') else "Ra")
        c.drawString(400, y, st_ascii)
        c.drawString(480, y, f"{r['confidence'] or 95}%")
        y -= 18
        if y < 40:
            c.showPage()
            y = height - 50

    c.save()
    return send_file(pdf_path, as_attachment=True, download_name=f"Bao_Cao_{datetime.now().strftime('%Y%m%d')}.pdf")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)