import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import mediapipe as mp

# Định nghĩa các lớp cử chỉ
classes = [
    'call', 'dislike', 'fist', 'four', 'mute', 'ok', 'one', 'palm',
    'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three',
    'two_up', 'two_up_inverted'
]
idx_to_class = {i: cls for i, cls in enumerate(classes)}

# Transform dữ liệu với tiền xử lý cải tiến
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load mô hình
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load('../model/best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Tăng ngưỡng để phát hiện rõ ràng hơn
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam!")
    exit()

frame_id = 0
print("\n🚀 Bắt đầu test webcam - Nhấn 'q' để thoát\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame!")
        break

    frame_id += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Điều chỉnh độ sáng và độ tương phản
    frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.2, beta=10)  # Tăng độ tương phản và sáng
    results = hands.process(frame_rgb)

    pred_label = "No hand"
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Tính bounding box lớn hơn
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand.landmark]
        y_coords = [lm.y * h for lm in hand.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Mở rộng vùng cắt để giống dữ liệu HaGRID
        margin = 50  # Tăng margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        # Cắt vùng tay
        hand_img = frame_rgb[y_min:y_max, x_min:x_max]
        if hand_img.size > 0:
            pil_img = Image.fromarray(hand_img)
            input_tensor = test_transform(pil_img).unsqueeze(0)

            # Dự đoán với ngưỡng tin cậy
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities.max().item()
                if confidence > 0.7:  # Chỉ chấp nhận dự đoán có độ tin cậy > 70%
                    pred_idx = output.argmax(dim=1).item()
                    pred_label = idx_to_class[pred_idx]
                else:
                    pred_label = "Uncertain"

            # Vẽ bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Hiển thị nhãn và độ tin cậy
    display_text = f"Pred: {pred_label} ({confidence:.2%})"
    cv2.putText(frame, display_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # In log
    print(f"[Frame {frame_id:04d}] Predicted: {pred_label} (Confidence: {confidence:.2%})")

    cv2.imshow('Hand Gesture Control - Demo cho GV', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()