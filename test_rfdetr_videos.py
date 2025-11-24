import cv2
import onnxruntime
import numpy as np
import random
import time

# ======================
# 工具函数
# ======================
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ======================
# 类别颜色固定
# ======================
class_color_map = {}
def get_color_for_class(class_id):
    if class_id not in class_color_map:
        h = random.randint(0, 179)
        s = random.randint(150, 255)
        v = random.randint(150, 255)
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        class_color_map[class_id] = tuple(int(x) for x in bgr[0][0])
    return class_color_map[class_id]

# ======================
# 图像预处理
# ======================
def preprocess_image(image, target_size=(560, 560)):
    h, w = image.shape[:2]
    w_rate = w / target_size[0]
    h_rate = h / target_size[1]

    resized_image = cv2.resize(image, target_size)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    normalized_image = rgb_image.astype(np.float32) / 255.0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized_image = (normalized_image - mean) / std

    processed_image = np.transpose(normalized_image, (2, 0, 1))[None, ...].astype(np.float32)
    return processed_image, (w_rate, h_rate)

# ======================
# 后处理
# ======================
def postprocess_detections(bboxes, labels, target_size=(560, 560)):
    results = []
    for i in range(len(bboxes)):
        bbox = xywh2xyxy(bboxes[i])
        bbox[0] *= target_size[1]
        bbox[2] *= target_size[1]
        bbox[1] *= target_size[0]
        bbox[3] *= target_size[0]

        label_list = sigmoid(labels[i])
        class_id = np.argmax(label_list)
        conf = label_list[class_id]

        results.append({
            'bbox': bbox,
            'class_id': class_id,
            'confidence': conf
        })
    return results

# ======================
# 绘制检测框
# ======================
def draw_detections(image, class_names, detections, scale_factors, conf_threshold=0.5):
    w_rate, h_rate = scale_factors
    for det in detections:
        conf = det['confidence']
        if conf < conf_threshold:
            continue

        bbox = det['bbox']
        class_id = det['class_id']
        color = get_color_for_class(class_id)

        x1 = int(bbox[0] * w_rate)
        y1 = int(bbox[1] * h_rate)
        x2 = int(bbox[2] * w_rate)
        y2 = int(bbox[3] * h_rate)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[class_id]} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

# ======================
# 视频/摄像头实时检测
# ======================
def run_video_detection(model_path, video_path, class_names, conf_threshold=0.5):
    print("加载 ONNX 模型中...")

    # 使用 GPU 推理，如果有 CUDA GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model_path, providers=providers)

    print("当前使用的推理 Provider:", session.get_providers())

    cap = cv2.VideoCapture(0 if video_path == "camera" else video_path)
    if not cap.isOpened():
        print("❌ 打开视频失败")
        return

    print("开始视频推理中，按 'q' 退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_image, scale_factors = preprocess_image(frame)

        start_time = time.time()
        output = session.run(["dets", "labels"], {"input": processed_image})
        infer_time = time.time() - start_time

        bboxes = output[0][0]
        labels = output[1][0]

        detections = postprocess_detections(bboxes, labels)
        result_frame = draw_detections(frame, class_names, detections, scale_factors, conf_threshold)

        fps = 1.0 / (infer_time + 1e-6)
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RF-DETR ONNX Video Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    model_path = "models/inference_model.onnx"

    # 视频检测：
    video_path = "test/车辆检测_2.mp4"
    # 摄像头检测：
    # video_path = "camera"

    class_names = ['ambulance', 'army vehicle','auto rickshaw','bicycle','bus','car',
                   'garbagevan','human hauler','minibus','minivan','motorbike','pickup',
                   'policecar','rickshaw','scooter','suv','taxi','three wheelers -CNG-',
                   'truck','van','wheelbarrow']

    run_video_detection(model_path, video_path, class_names)

