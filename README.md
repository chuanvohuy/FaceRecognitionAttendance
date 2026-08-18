# 📷 Hệ Thống Điểm Danh Bằng Nhận Diện Khuôn Mặt

Mô tả ngắn gọn: Hệ thống hỗ trợ điểm danh nhân viên/sinh viên tự động qua Webcam sử dụng Python và thư viện OpenCV.

---

## I. Giới thiệu đề tài

### Lý do chọn đề tài
* **Hiện trạng:** Việc điểm danh truyền thống bằng giấy hoặc điểm danh thủ công tốn nhiều thời gian và dễ xảy ra tình trạng điểm danh hộ.
* **Hạn chế thực tế:** Các hệ thống thẻ từ hoặc vân tay tốn chi phí thiết bị và không đảm bảo tính không tiếp xúc.
* **Giải pháp:** Xây dựng hệ thống tự động nhận diện khuôn mặt qua camera, giúp tối ưu thời gian và đảm bảo độ chính xác cao.

---

## II. Yêu cầu chức năng

| STT | Chức năng | Mô tả |
| :--- | :--- | :--- |
| 1 | Quản lý thông tin | Thêm, sửa, xóa thông tin nhân viên/sinh viên và lưu trữ dữ liệu khuôn mặt. |
| 2 | Nhận diện Realtime | Mở camera quét và nhận diện khuôn mặt theo thời gian thực. |
| 3 | Ghi nhận điểm danh | Tự động lưu lịch sử gồm Mã NV, Họ tên và Thời gian check-in. |
| 4 | Xuất báo cáo | Xuất dữ liệu điểm danh ra file Excel/CSV. |

---

## III. Công cụ và công nghệ sử dụng

| Thành phần | Công nghệ |
| :--- | :--- |
| Ngôn ngữ lập trình | Python 3.11 |
| Thư viện AI / Xử lý ảnh | OpenCV, face_recognition, NumPy |
| Cơ sở dữ liệu | SQLite / MySQL |
| Quản lý mã nguồn | Git & GitHub |

---

## IV. Thành viên nhóm

| STT | Họ và tên | MSSV | Vai trò | Phân Công | GitHub |
| :---: | :--- | :---: | :--- | :--- | :--- |
| 1 | **Võ Huy Nguyên Chuẩn** | 24108063 | Trưởng Nhóm | Lập trình xử lý hình ảnh (Python Lead) | [...] |
| 2 | **Vũ Việt Dũng** | 24107630 | Thành Viên | Lập trình CSDL & Tích hợp hệ thống | [...] |
| 3 | **Vũ Tiến Đạt** | 24108659 | Thành Viên | Xây dựng giao diện & Báo cáo | [...] |
