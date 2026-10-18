import sqlite3
import os
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), 'face_checkin.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def add_col_if_missing(cursor, table, col_def):
    col_name = col_def.split()[0]
    cursor.execute(f"PRAGMA table_info({table})")
    existing_cols = [c[1] for c in cursor.fetchall()]
    if col_name not in existing_cols:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # 1. Bảng Phòng Ban (Departments)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT
        )
    ''')

    # 2. Bảng Ca Làm Việc (Shifts)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shifts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            late_grace_minutes INTEGER DEFAULT 15,
            early_leave_minutes INTEGER DEFAULT 15
        )
    ''')
    add_col_if_missing(cursor, 'shifts', 'standard_hours REAL DEFAULT 8.0')
    add_col_if_missing(cursor, 'shifts', 'rate_multiplier REAL DEFAULT 1.0')

    # 3. Bảng Nhân Viên (Employees)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_code TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            shift_start TEXT NOT NULL DEFAULT '08:00',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    add_col_if_missing(cursor, 'employees', 'email TEXT')
    add_col_if_missing(cursor, 'employees', 'phone TEXT')
    add_col_if_missing(cursor, 'employees', 'department_id INTEGER')
    add_col_if_missing(cursor, 'employees', 'position TEXT DEFAULT "Nhân viên"')
    add_col_if_missing(cursor, 'employees', 'shift_id INTEGER')
    add_col_if_missing(cursor, 'employees', 'status TEXT DEFAULT "Đang làm việc"')
    add_col_if_missing(cursor, 'employees', 'avatar_path TEXT')
    add_col_if_missing(cursor, 'employees', 'face_registered INTEGER DEFAULT 0')
    add_col_if_missing(cursor, 'employees', 'base_salary REAL DEFAULT 8000000.0')

    # 4. Bảng Tài Khoản (Users)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'staff',
            full_name TEXT NOT NULL
        )
    ''')
    add_col_if_missing(cursor, 'users', 'emp_id INTEGER')
    add_col_if_missing(cursor, 'users', 'locked_until TIMESTAMP')
    add_col_if_missing(cursor, 'users', 'plain_password TEXT')
    add_col_if_missing(cursor, 'users', 'failed_attempts INTEGER DEFAULT 0')
    add_col_if_missing(cursor, 'users', 'unlocked_by TEXT')

    # 5. Bảng Điểm Danh (Attendance)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id INTEGER,
            emp_name TEXT NOT NULL,
            status TEXT NOT NULL,
            snapshot_path TEXT
        )
    ''')
    add_col_if_missing(cursor, 'attendance', 'check_date TEXT')
    add_col_if_missing(cursor, 'attendance', 'check_time TEXT')
    add_col_if_missing(cursor, 'attendance', 'checkin_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
    add_col_if_missing(cursor, 'attendance', 'check_type TEXT DEFAULT "Chấm công Vào"')
    add_col_if_missing(cursor, 'attendance', 'confidence REAL DEFAULT 95.0')
    add_col_if_missing(cursor, 'attendance', 'liveness_status TEXT DEFAULT "Verified"')
    add_col_if_missing(cursor, 'attendance', 'work_hours REAL DEFAULT 0.0')
    add_col_if_missing(cursor, 'attendance', 'overtime_hours REAL DEFAULT 0.0')
    add_col_if_missing(cursor, 'attendance', 'device_code TEXT DEFAULT "KIOSK-MAIN"')
    add_col_if_missing(cursor, 'attendance', 'is_anomaly INTEGER DEFAULT 0')
    add_col_if_missing(cursor, 'attendance', 'anomaly_reason TEXT')
    add_col_if_missing(cursor, 'attendance', 'approved_by TEXT')
    add_col_if_missing(cursor, 'attendance', 'offline_synced INTEGER DEFAULT 0')
    add_col_if_missing(cursor, 'attendance', 'sync_hash TEXT')

    # 6. Bảng Nhật Ký Hoạt Động (Audit Logs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            detail TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 7. Bảng Lập Lịch Ca Làm Việc (Work Schedules)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS work_schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id INTEGER NOT NULL,
            shift_id INTEGER NOT NULL,
            work_date TEXT NOT NULL,
            note TEXT,
            assigned_by TEXT DEFAULT 'admin',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(emp_id, work_date)
        )
    ''')

    # 8. Bảng Đơn Từ & Yêu Cầu (Requests: Đổi ca, Xin nghỉ, Tăng ca, Khiếu nại công)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id INTEGER NOT NULL,
            request_type TEXT NOT NULL,
            title TEXT NOT NULL,
            reason TEXT NOT NULL,
            target_date TEXT NOT NULL,
            new_shift_id INTEGER,
            overtime_hours REAL DEFAULT 0.0,
            attendance_id INTEGER,
            status TEXT DEFAULT 'Chờ duyệt',
            approver TEXT,
            approver_note TEXT,
            approved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 9. Bảng Thiết Bị Kiosk Điểm Danh (Kiosks)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kiosks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            status TEXT DEFAULT 'online',
            ip_address TEXT,
            last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 10. Bảng Thông Báo Thời Gian Thực (Notifications)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipient_role TEXT DEFAULT 'all',
            recipient_id INTEGER,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            type TEXT DEFAULT 'info',
            is_read INTEGER DEFAULT 0,
            link TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 11. Bảng Cài Đặt Hệ Thống (System Settings)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            description TEXT
        )
    ''')

    # Khởi tạo cài đặt mặc định nếu chưa có
    default_settings = [
        ('min_confidence', '78.0', 'Ngưỡng nhận diện khuôn mặt (%)'),
        ('max_retry_attempts', '3', 'Số lần quét sai tối đa trước khi tạm khóa'),
        ('lockout_duration_mins', '2', 'Thời gian tạm khóa (phút)'),
        ('enable_iqa', '1', 'Bật kiểm tra chất lượng ảnh (độ sáng, mờ, góc chụp)'),
        ('iqa_min_blur', '60.0', 'Ngưỡng chống ảnh mờ (Laplacian variance)'),
        ('iqa_min_brightness', '35.0', 'Ngưỡng độ sáng tối thiểu (0-255)'),
        ('auto_backup_enabled', '1', 'Tự động tạo bản sao lưu cơ sở dữ liệu')
    ]
    for k, v, desc in default_settings:
        cursor.execute("INSERT OR IGNORE INTO system_settings (key, value, description) VALUES (?, ?, ?)", (k, v, desc))

    # Cập nhật dữ liệu cũ đồng bộ
    cursor.execute("UPDATE employees SET status = 'Đang làm việc' WHERE status IS NULL")
    cursor.execute("UPDATE attendance SET check_date = date('now') WHERE check_date IS NULL")
    cursor.execute("UPDATE attendance SET check_time = time('now') WHERE check_time IS NULL")

    # Dữ liệu phòng ban mặc định
    cursor.execute("SELECT COUNT(*) FROM departments")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO departments (name, description) VALUES ('Phòng Kỹ Thuật (IT)', 'Phát triển phần mềm & Hạ tầng')")
        cursor.execute("INSERT INTO departments (name, description) VALUES ('Phòng Kế Toán', 'Tài chính và Kế toán')")
        cursor.execute("INSERT INTO departments (name, description) VALUES ('Phòng Nhân Sự (HR)', 'Quản trị nhân lực')")

    # Dữ liệu ca làm mặc định
    cursor.execute("SELECT COUNT(*) FROM shifts")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO shifts (name, start_time, end_time, late_grace_minutes, standard_hours, rate_multiplier) VALUES ('Ca Sáng', '08:00', '12:00', 15, 4.0, 1.0)")
        cursor.execute("INSERT INTO shifts (name, start_time, end_time, late_grace_minutes, standard_hours, rate_multiplier) VALUES ('Ca Chiều', '13:30', '17:30', 15, 4.0, 1.0)")
        cursor.execute("INSERT INTO shifts (name, start_time, end_time, late_grace_minutes, standard_hours, rate_multiplier) VALUES ('Ca Hành Chính', '08:00', '17:30', 15, 8.0, 1.0)")

    # Kiosk mặc định
    cursor.execute("SELECT COUNT(*) FROM kiosks")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO kiosks (device_code, name, location, status, ip_address) VALUES ('KIOSK-MAIN', 'Kiosk Sảnh Chính', 'Tầng 1 - Cổng Ra Vào', 'online', '127.0.0.1')")
        cursor.execute("INSERT INTO kiosks (device_code, name, location, status, ip_address) VALUES ('KIOSK-TECH', 'Kiosk Khối Kỹ Thuật', 'Tầng 3 - Phòng Lab AI', 'online', '192.168.1.105')")

    # Tài khoản quản trị mặc định
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO users (username, password, role, full_name, plain_password)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', generate_password_hash('admin123'), 'admin', 'Quản trị viên Hệ thống', 'admin123'))

    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'manager'")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO users (username, password, role, full_name, plain_password)
            VALUES (?, ?, ?, ?, ?)
        ''', ('manager', generate_password_hash('manager123'), 'manager', 'Quản lý Nhân sự', 'manager123'))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database FaceCheckIn Enterprise đã được nâng cấp toàn diện và đồng bộ thành công.")