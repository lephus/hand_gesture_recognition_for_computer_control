# Hướng Dẫn Test Model Hand Gesture Recognition

## 📋 Tổng Quan
Tài liệu này hướng dẫn cách test model nhận diện cử chỉ tay đã được train từ Kaggle.

## 🚀 Bước 1: Thiết Lập Môi Trường

### 1.1 Tạo Virtual Environment
```bash
# Di chuyển vào thư mục project
cd /Users/lehuuphu/Downloads/DUT-ths/ComputerVision/hand_gesture_recognition_for_computer_control

# Tạo virtual environment
python3 -m venv venv

# Kích hoạt virtual environment
# Trên macOS/Linux:
source venv/bin/activate
# Trên Windows:
# venv\Scripts\activate
```

### 1.2 Cài Đặt Dependencies
```bash
# Cài đặt các package cần thiết
cd backend
pip install -r requirements.txt

# Hoặc cài đặt từng package nếu có lỗi:
pip install tensorflow==2.13.0
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-socketio==5.10.0
pip install pyautogui==0.9.54
pip install pynput==1.7.6
```

## 🧪 Bước 2: Test Model

### 2.1 Test Cơ Bản (Không cần webcam)
```bash
cd backend
python test_model.py
# Chọn option 1: Basic model test
```

### 2.2 Test Với Sample Images
```bash
cd backend
python test_model.py
# Chọn option 2: Test with sample images
```

### 2.3 Test Với Webcam (Real-time)
```bash
cd backend
python test_model.py
# Chọn option 3: Test with webcam (live)
```

### 2.4 Test Tất Cả
```bash
cd backend
python test_model.py
# Chọn option 4: All tests
```

## 🌐 Bước 3: Chạy Backend Server

### 3.1 Khởi Động Server
```bash
cd backend
python main.py
```

Server sẽ chạy trên: `http://localhost:5000`

### 3.2 Test API Endpoints
```bash
# Health check
curl http://localhost:5000/health

# Get config
curl http://localhost:5000/api/config

# Get stats
curl http://localhost:5000/api/stats
```

## 📊 Bước 4: Test WebSocket Connection

### 4.1 Sử dụng JavaScript Client
Tạo file `test_client.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Gesture Test Client</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
</head>
<body>
    <h1>Gesture Recognition Test</h1>
    <video id="video" width="640" height="480" autoplay></video>
    <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
    <div id="results"></div>
    
    <script>
        const socket = io('http://localhost:5000');
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const results = document.getElementById('results');
        
        // Get webcam access
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.play();
            })
            .catch(err => console.error('Error accessing webcam:', err));
        
        // Send frames to server
        setInterval(() => {
            ctx.drawImage(video, 0, 0, 640, 480);
            const frame = canvas.toDataURL('image/jpeg');
            socket.emit('frame', { frame: frame });
        }, 100); // Send every 100ms
        
        // Listen for results
        socket.on('gesture_result', (data) => {
            results.innerHTML = `
                <h3>Gesture: ${data.gesture}</h3>
                <p>Confidence: ${data.confidence.toFixed(3)}</p>
                <p>Inference Time: ${data.inference_time.toFixed(1)}ms</p>
            `;
        });
        
        socket.on('error', (data) => {
            console.error('Error:', data.message);
        });
    </script>
</body>
</html>
```

## 🔧 Troubleshooting

### Lỗi Thường Gặp:

#### 1. Model không load được
```bash
# Kiểm tra file model có tồn tại không
ls -la backend/models/best_model_v1.h5

# Kiểm tra quyền truy cập
chmod 644 backend/models/best_model_v1.h5
```

#### 2. TensorFlow không cài được
```bash
# Cài đặt TensorFlow phiên bản khác
pip install tensorflow==2.12.0
# hoặc
pip install tensorflow==2.14.0
```

#### 3. OpenCV không hoạt động
```bash
# Cài đặt lại OpenCV
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

#### 4. Webcam không hoạt động
```bash
# Kiểm tra webcam có được sử dụng bởi app khác không
# Đóng các ứng dụng khác đang sử dụng webcam
```

### Kiểm Tra Dependencies:
```bash
# Kiểm tra Python version
python --version

# Kiểm tra TensorFlow
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Kiểm tra OpenCV
python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Kiểm tra model file
python -c "import tensorflow as tf; model = tf.keras.models.load_model('backend/models/best_model_v1.h5'); print('Model loaded successfully')"
```

## 📈 Đánh Giá Hiệu Suất

### Metrics Quan Trọng:
- **Inference Time**: Thời gian xử lý mỗi frame (ms)
- **FPS**: Số frame xử lý mỗi giây
- **Accuracy**: Độ chính xác nhận diện
- **Confidence**: Độ tin cậy của prediction

### Gesture Classes Được Hỗ Trợ:
- `one_finger`: 1 ngón tay
- `two_fingers`: 2 ngón tay  
- `three_fingers`: 3 ngón tay
- `four_fingers`: 4 ngón tay
- `five_fingers`: 5 ngón tay
- `neutral`: Không có cử chỉ
- `rotate_clockwise`: Xoay theo chiều kim đồng hồ
- `rotate_counterclockwise`: Xoay ngược chiều kim đồng hồ
- `x_gesture`: Cử chỉ chữ X
- `swipe_left`: Vuốt trái
- `swipe_right`: Vuốt phải

## 🎯 Kết Quả Mong Đợi

### Test Thành Công:
- Model load được không có lỗi
- Inference time < 100ms
- Confidence > 0.8 cho các gesture rõ ràng
- FPS > 10 frames/second

### Nếu Có Vấn Đề:
1. Kiểm tra lại model file
2. Cài đặt lại dependencies
3. Kiểm tra webcam permissions
4. Test với sample images trước

## 📞 Hỗ Trợ

Nếu gặp lỗi, hãy chạy lệnh sau để gửi thông tin debug:
```bash
cd backend
python -c "
import sys
print('Python version:', sys.version)
try:
    import tensorflow as tf
    print('TensorFlow version:', tf.__version__)
except ImportError as e:
    print('TensorFlow error:', e)
try:
    import cv2
    print('OpenCV version:', cv2.__version__)
except ImportError as e:
    print('OpenCV error:', e)
"
```
