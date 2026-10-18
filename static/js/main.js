document.addEventListener('DOMContentLoaded', async () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('snapshotCanvas');
    const captureBtn = document.getElementById('captureBtn');
    const autoCheckin = document.getElementById('autoCheckin');
    const liveClock = document.getElementById('liveClock');

    // Liveness HUD Elements
    const livenessHud = document.getElementById('livenessHud');
    const livenessBadgeText = document.getElementById('livenessBadgeText');
    const challengeIcon = document.getElementById('challengeIcon');
    const challengeTitle = document.getElementById('challengeTitle');
    const challengeDesc = document.getElementById('challengeDesc');
    const livenessProgressWrap = document.getElementById('livenessProgressWrap');
    const livenessProgressBar = document.getElementById('livenessProgressBar');

    // Result Modal Elements
    const resultModal = document.getElementById('resultModal');
    const resName = document.getElementById('resName');
    const resTimeType = document.getElementById('resTimeType');
    const resStatusBadge = document.getElementById('resStatusBadge');
    const resConfidence = document.getElementById('resConfidence');
    const resLivenessBadge = document.getElementById('resLivenessBadge');
    const resIcon = document.getElementById('resIcon');

    let currentCheckType = 'Chấm công Vào';

    window.setCheckType = function(type) {
        currentCheckType = type;
        const bIn = document.getElementById('btnCheckIn');
        const bOut = document.getElementById('btnCheckOut');
        if (bIn) bIn.classList.toggle('active', type === 'Chấm công Vào');
        if (bOut) bOut.classList.toggle('active', type === 'Chấm công Ra');
    };

    if (!video) return;

    // 1. Đồng hồ thời gian thực
    setInterval(() => {
        const now = new Date();
        if (liveClock) liveClock.innerText = now.toLocaleTimeString('vi-VN');
    }, 1000);

    // 2. Kiosk Heartbeat (Gửi tín hiệu trực tuyến mỗi 45 giây)
    setInterval(async () => {
        try {
            await fetch('/api/kiosk/heartbeat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ device_code: 'KIOSK-MAIN' })
            });
        } catch(e) {}
    }, 45000);

    // 3. Tự động đồng bộ ngoại tuyến khi có mạng trở lại (Offline Mode Sync)
    async function syncOfflineRecords() {
        const cached = JSON.parse(localStorage.getItem('offline_checkins') || '[]');
        if (cached.length === 0) return;

        try {
            const res = await fetch('/api/checkin/offline_sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ records: cached })
            });
            const d = await res.json();
            if (d.success) {
                localStorage.removeItem('offline_checkins');
                Swal.fire({
                    toast: true, position: 'top-end', icon: 'info',
                    title: 'Đồng bộ ngoại tuyến', text: d.message,
                    showConfirmButton: false, timer: 3000, background: '#0f172a', color: '#fff'
                });
            }
        } catch(e) {}
    }
    window.addEventListener('online', syncOfflineRecords);
    syncOfflineRecords(); // Thử đồng bộ ngay khi tải trang

    // 4. Âm thanh chuông báo Web Audio
    function playChime(type = 'success') {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';

            if (type === 'success') {
                osc.frequency.setValueAtTime(1046.50, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(1567.98, ctx.currentTime + 0.12);
                gain.gain.setValueAtTime(0.18, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
            } else if (type === 'challenge') {
                osc.frequency.setValueAtTime(880, ctx.currentTime);
                osc.frequency.setValueAtTime(1174.66, ctx.currentTime + 0.1);
                gain.gain.setValueAtTime(0.12, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
            } else {
                osc.frequency.setValueAtTime(320, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(180, ctx.currentTime + 0.25);
                gain.gain.setValueAtTime(0.2, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
            }

            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.45);
        } catch (e) {}
    }

    // 5. Quản lý trạng thái giao diện HUD Thử Thách
    function setHudState(state, title, desc, iconClass) {
        if (!livenessHud) return;
        livenessHud.classList.remove('challenging', 'success', 'failed');
        if (state) livenessHud.classList.add(state);
        if (title && challengeTitle) challengeTitle.innerText = title;
        if (desc && challengeDesc) challengeDesc.innerText = desc;
        if (iconClass && challengeIcon) challengeIcon.className = iconClass;
    }

    // 6. Khởi tạo Camera tương thích an toàn
    async function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            Swal.fire({
                icon: 'warning',
                title: 'Bảo Mật Trình Duyệt',
                html: `Trình duyệt chặn Camera do bạn đang truy cập qua IP HTTP: <b>${window.location.host}</b>.<br><br>👉 Vui lòng mở bằng: <a href="http://localhost:5000/checkin" style="color:#38bdf8;font-weight:bold;">http://localhost:5000/checkin</a>`,
                confirmButtonText: 'Đã hiểu'
            });
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false
            });
            video.srcObject = stream;
        } catch (err) {
            try {
                const fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video.srcObject = fallbackStream;
            } catch (fallbackErr) {
                Swal.fire({
                    icon: 'error',
                    title: 'Lỗi Truy Cập Camera',
                    html: `Vui lòng bấm vào <b>biểu tượng ổ khóa/cài đặt</b> bên trái thanh địa chỉ trình duyệt và chuyển <b>Camera sang "Cho phép (Allow)"</b> rồi tải lại trang!`,
                    confirmButtonText: 'Đóng'
                });
            }
        }
    }

    startCamera();

    // 7. Xử lý Chấm Công & Chống Giả Mạo Liveness Detection & IQA
    let isProcessing = false;
    async function processVerification() {
        if (isProcessing) return;
        if (!video.videoWidth || !video.videoHeight) return;

        isProcessing = true;

        try {
            // Bước 1: Yêu cầu mã thử thách ngẫu nhiên từ máy chủ (Chống replay attack)
            setHudState('', 'Đang khởi tạo thử thách...', 'Máy chủ đang chọn chỉ dẫn ngẫu nhiên...', 'fa-solid fa-spinner fa-spin');
            const challengeRes = await fetch('/api/liveness/challenge');
            const challengeData = await challengeRes.json();

            if (!challengeData.success || !challengeData.challenge) {
                setHudState('failed', 'Lỗi kết nối AI', 'Không nhận được thử thách từ máy chủ.', 'fa-solid fa-triangle-exclamation');
                isProcessing = false;
                return;
            }

            const challenge = challengeData.challenge;

            // Bước 2: Hiển thị chỉ dẫn thử thách cho người dùng
            playChime('challenge');
            setHudState(
                'challenging',
                `👉 ${challenge.title.toUpperCase()}`,
                challenge.instruction,
                `${challenge.icon} fa-bounce`
            );

            if (livenessProgressWrap) {
                livenessProgressWrap.style.display = 'block';
                if (livenessProgressBar) livenessProgressBar.style.width = '0%';
            }

            // Đợi 400ms để người dùng đọc và chuẩn bị thực hiện hành động
            await new Promise(r => setTimeout(r, 400));

            // Bước 3: Thu thập chuỗi 6 khung hình trong quá trình người dùng chuyển động
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');

            const frames = [];
            const totalFrames = 6;
            const frameDelay = 220; // 220ms mỗi frame (~1.3 giây tổng chuỗi cử động)

            for (let i = 0; i < totalFrames; i++) {
                ctx.save();
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                ctx.restore();

                const base64Frame = canvas.toDataURL('image/jpeg', 0.85);
                frames.push(base64Frame);

                const percent = Math.round(((i + 1) / totalFrames) * 100);
                if (livenessProgressBar) livenessProgressBar.style.width = `${percent}%`;

                if (i < totalFrames - 1) {
                    await new Promise(r => setTimeout(r, frameDelay));
                }
            }

            // Bước 4: Gửi chuỗi khung hình lên Backend để phân tích IQA, Liveness & Face ID
            setHudState('challenging', 'Đang phân tích Liveness & IQA...', 'Đối soát chuỗi khung hình & nhận diện...', 'fa-solid fa-brain fa-pulse');

            const response = await fetch('/api/checkin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    frames: frames,
                    challenge_type: challenge.type,
                    challenge_token: challenge.token,
                    check_type: currentCheckType,
                    device_code: 'KIOSK-MAIN'
                })
            });

            const data = await response.json();

            // TRƯỜNG HỢP 1: LỖI CHẤT LƯỢNG ẢNH (IQA FAILED)
            if (data.iqa_failed) {
                playChime('error');
                setHudState('failed', 'Chất lượng ảnh chưa đạt', data.message, 'fa-solid fa-circle-exclamation');
                Swal.fire({
                    icon: 'warning',
                    title: 'Chất Lượng Ảnh Chưa Đạt',
                    html: `<b>${data.message}</b><br><br><small style="color:#94a3b8;">Vui lòng điều chỉnh vị trí đứng và giữ thẳng camera để hệ thống nhận diện chuẩn xác nhất.</small>`,
                    background: '#0f172a', color: '#fff', confirmButtonColor: '#2563eb'
                });

            // TRƯỜNG HỢP 2: THẤT BẠI DO KHÔNG VƯỢT QUA LIVENESS
            } else if (data.liveness_failed) {
                playChime('error');
                setHudState('failed', 'Liveness Thất Bại', data.message, 'fa-solid fa-user-xmark');
                Swal.fire({
                    icon: 'warning',
                    title: 'Xác Minh Người Thật Thất Bại',
                    html: `<b>${data.message}</b><br><br><span style="color:#94a3b8;font-size:0.85rem;">Hệ thống yêu cầu cử động thực tế theo chỉ dẫn và từ chối ảnh tĩnh hoặc video ghi hình sẵn.</span>`,
                    confirmButtonText: 'Đã hiểu & Thử lại',
                    background: '#0f172a', color: '#fff', confirmButtonColor: '#2563eb'
                });

            // TRƯỜNG HỢP 3: NHẬN DIỆN CHƯA KHỚP & CHO PHÉP THỬ LẠI (RETRY POLICY)
            } else if (data.retry_allowed) {
                playChime('error');
                setHudState('failed', 'Chưa khớp khuôn mặt', data.message, 'fa-solid fa-triangle-exclamation');
                Swal.fire({
                    toast: true, position: 'top-end', icon: 'warning',
                    title: 'Nhận diện chưa khớp',
                    text: data.message,
                    showConfirmButton: false, timer: 4000, background: '#0f172a', color: '#fff'
                });

            // TRƯỜNG HỢP 4: THÀNH CÔNG VƯỢT QUA TẤT CẢ TẦNG KIỂM TRA
            } else if (data.success) {
                playChime('success');
                setHudState('success', `✅ Đã xác minh: ${data.name}`, `${data.time} — ${data.check_type}`, 'fa-solid fa-circle-check');

                if (resultModal) {
                    resultModal.style.display = 'flex';
                    resIcon.innerHTML = '<i class="fa-solid fa-circle-check" style="color:#10b981;"></i>';
                    resName.innerText = `✅ ${data.name}`;
                    resTimeType.innerText = `${data.time} — ${data.check_type}`;
                    resStatusBadge.className = `pill ${data.status === 'Đúng giờ' ? 'pill-success' : 'pill-danger'}`;
                    resStatusBadge.innerText = data.status;
                    resConfidence.innerText = `Confidence ${data.confidence}%`;
                    if (resLivenessBadge) {
                        resLivenessBadge.className = 'pill pill-success';
                        resLivenessBadge.innerHTML = '<i class="fa-solid fa-shield-check"></i> Liveness Passed';
                    }
                }

                Swal.fire({
                    toast: true, position: 'top-end', icon: 'success',
                    title: `${data.name} — ${data.status}`,
                    text: `${data.check_type} (${data.confidence}%) • Liveness OK`,
                    showConfirmButton: false, timer: 3500, background: '#0f172a', color: '#fff'
                });

            // TRƯỜNG HỢP 5: THẤT BẠI 3 LẦN LIÊN TIẾP -> TẠM KHÓA TÀI KHOẢN
            } else if (data.kicked) {
                playChime('error');
                setHudState('failed', 'Tài khoản bị tạm khóa', data.message, 'fa-solid fa-lock');
                if (autoCheckin) autoCheckin.checked = false;

                Swal.fire({
                    icon: 'error',
                    title: 'Tài Khoản Bị Tạm Khóa',
                    html: `<b>${data.message}</b><br><br><span style="color:#94a3b8;">Vui lòng liên hệ Quản trị viên để được mở khóa hoặc xác minh thủ công.</span>`,
                    showConfirmButton: false, timer: 3500, background: '#0f172a', color: '#fff'
                });
                setTimeout(() => { window.location.href = '/staff/login'; }, 3500);
                return;

            // TRƯỜNG HỢP 6: ĐÃ CHẤM CÔNG RỒI
            } else if (data.already_checked) {
                playChime('error');
                setHudState('success', 'Đã Hoàn Tất Hôm Nay', data.message, 'fa-solid fa-calendar-check');
                Swal.fire({
                    toast: true, position: 'top-end', icon: 'warning',
                    title: 'Thông báo', text: data.message,
                    showConfirmButton: false, timer: 3500, background: '#0f172a', color: '#fff'
                });
            } else {
                setHudState('failed', 'Không nhận diện được', data.message, 'fa-solid fa-triangle-exclamation');
            }

        } catch (e) {
            console.error('Checkin Network Error:', e);
            // Xử lý lưu ngoại tuyến (Offline Mode)
            if (!navigator.onLine) {
                const now = new Date();
                const offlineRecord = {
                    time: now.toLocaleTimeString('vi-VN'),
                    check_date: now.toISOString().slice(0, 10),
                    check_time: now.toTimeString().slice(0, 8),
                    checkin_time: now.toISOString().replace('T', ' ').slice(0, 19),
                    check_type: currentCheckType,
                    status: 'Đúng giờ',
                    confidence: 95.0
                };
                const cached = JSON.parse(localStorage.getItem('offline_checkins') || '[]');
                cached.push(offlineRecord);
                localStorage.setItem('offline_checkins', JSON.stringify(cached));

                playChime('success');
                setHudState('success', 'Đã Lưu Ngoại Tuyến', 'Mất mạng: Bản ghi đã lưu tạm và sẽ tự động đồng bộ khi có kết nối!', 'fa-solid fa-wifi');
                Swal.fire({
                    icon: 'info',
                    title: 'Chế Độ Ngoại Tuyến (Offline)',
                    text: 'Mất kết nối mạng. Điểm danh đã được lưu tạm trên thiết bị và sẽ tự động đồng bộ khi có mạng trở lại!',
                    background: '#0f172a', color: '#fff', confirmButtonColor: '#2563eb'
                });
            } else {
                setHudState('failed', 'Lỗi Hệ Thống', 'Không thể kết nối đến máy chủ AI.', 'fa-solid fa-circle-exclamation');
            }
        } finally {
            if (livenessProgressWrap) {
                setTimeout(() => { livenessProgressWrap.style.display = 'none'; }, 1000);
            }
            setTimeout(() => {
                isProcessing = false;
                setHudState('', 'Sẵn sàng nhận diện', 'Đứng thẳng trước camera để bắt đầu thử thách', 'fa-solid fa-shield-halved');
            }, 3500);
        }
    }

    if (captureBtn) captureBtn.addEventListener('click', () => processVerification());

    setInterval(() => {
        if (autoCheckin && autoCheckin.checked && !isProcessing) {
            processVerification();
        }
    }, 5500);
});