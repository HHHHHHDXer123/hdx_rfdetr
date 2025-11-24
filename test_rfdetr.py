import cv2
import onnxruntime
import numpy as np
import random
import time
import torchvision.transforms.functional as F

def xywh2xyxy(x):
    """
    将边界框坐标从 (x, y, width, height) 转换为 (x1, y1, x2, y2)

    Args:
        x (np.ndarray): 输入边界框数组

    Returns:
        np.ndarray: 转换后的边界框数组
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def softmax(x):
    """
    Softmax函数

    Args:
        x (np.ndarray): 输入数组

    Returns:
        np.ndarray: 经过softmax处理的数组
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    """
    Sigmoid函数
    Args:
        x (np.ndarray): 输入数组
    Returns:
        np.ndarray: 经过sigmoid处理的数组
    """
    return 1 / (1 + np.exp(-x))


def get_optimal_font_scale(image_shape, text, font_face=cv2.FONT_HERSHEY_SIMPLEX):
    """
    根据图像分辨率自动计算最优字体大小
    """
    # 获取图像的最小边长
    min_dimension = min(image_shape[0], image_shape[1])

    # 基于图像尺寸计算基础字体大小
    base_font_scale = min_dimension / 1000.0

    # 确保字体大小在合理范围内
    font_scale = max(0.5, min(base_font_scale, 2.0))

    return font_scale


def generate_distinct_color():
    """
    生成具有明显区分度的随机颜色
    """
    # 使用HSV色彩空间生成颜色，确保颜色具有较高的饱和度和亮度
    h = random.randint(0, 179)  # OpenCV中色调范围是0-179
    s = random.randint(150, 255)  # 饱和度：150-255 (避免过淡的颜色)
    v = random.randint(150, 255)  # 亮度：150-255 (避免过暗的颜色)

    # 将HSV转换为BGR
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0][0])


def get_color_for_class(class_id, color_map):
    """
    为类别ID获取颜色，如果不存在则生成新颜色
    """
    if class_id not in color_map:
        color_map[class_id] = generate_distinct_color()
    return color_map[class_id]


def preprocess_image(image, target_size=(560, 560)):
    """
    预处理图像
    Args:
        image (np.ndarray): 输入图像
        target_size (tuple): 目标图像尺寸
    Returns:
        tuple: (处理后的图像, 缩放因子, 原始尺寸)
    """
    # 保存原始尺寸
    h, w = image.shape[:2]
    w_rate = w / target_size[0]
    h_rate = h / target_size[1]

    # 调整图像大小
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # 颜色空间转换和归一化
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    normalized_image = rgb_image.astype(np.float32) / 255.0

    # 标准化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized_image = (normalized_image - mean) / std

    # 转换维度并扩展批次维度
    processed_image = np.transpose(normalized_image, (2, 0, 1))
    processed_image = np.expand_dims(processed_image, axis=0).astype(np.float32)

    return processed_image, (w_rate, h_rate), (w, h)


def preprocess_image_torch(image, target_size=(560, 560)):
    """
    预处理图像
    Args:
        image (np.ndarray): 输入图像
        target_size (tuple): 目标图像尺寸
    Returns:
        tuple: (处理后的图像, 缩放因子, 原始尺寸)
    """
    img_tensor = image
    img_tensor = F.to_tensor(img_tensor).to("cuda:0")

    h, w = img_tensor.shape[1:]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_tensor = F.normalize(img_tensor, mean, std)
    img_tensor = F.resize(img_tensor, target_size)

    # 保存原始尺寸
    h, w = image.shape[:2]
    w_rate = w / target_size[0]
    h_rate = h / target_size[1]

    return np.expand_dims(img_tensor.cpu().numpy(), axis=0).astype(np.float32), (w_rate, h_rate), (w, h)


def postprocess_detections(bboxes, labels, target_size=(560, 560)):
    """
    后处理检测结果

    Args:
        bboxes (np.ndarray): 边界框数组
        labels (np.ndarray): 标签数组
        target_size (tuple): 目标图像尺寸

    Returns:
        list: 处理后的检测结果列表
    """
    results = []
    for i in range(len(bboxes)):
        # 转换边界框格式
        bbox = xywh2xyxy(bboxes[i])

        # 缩放边界框坐标
        bbox[0] *= target_size[1]  # x坐标乘以宽度
        bbox[2] *= target_size[1]  # x坐标乘以宽度
        bbox[1] *= target_size[0]  # y坐标乘以高度
        bbox[3] *= target_size[0]  # y坐标乘以高度

        # 应用softmax并获取类别和置信度
        label_list = sigmoid(labels[i])
        class_id = np.argmax(label_list)
        conf = label_list[class_id]

        results.append({
            'bbox': bbox,
            'class_id': class_id,
            'confidence': conf
        })

    return results


def draw_detections(image, class_names, detections, scale_factors, conf_threshold=0.5):
    """
    在图像上绘制检测结果

    Args:
        image (np.ndarray): 输入图像
        detections (list): 检测结果列表
        scale_factors (tuple): 缩放因子 (w_rate, h_rate)
        conf_threshold (float): 置信度阈值

    Returns:
        np.ndarray: 绘制了检测结果的图像
    """
    w_rate, h_rate = scale_factors
    result_image = image.copy()
    detection_count = 0

    # 存储类别颜色的字典
    color_map = {}

    # 根据图像分辨率自动调整字体大小
    font_scale = get_optimal_font_scale(image.shape, "SampleText")
    thickness = max(1, int(font_scale * 2))  # 根据字体大小调整线条粗细

    for detection in detections:
        conf = detection['confidence']
        if conf > conf_threshold:
            detection_count += 1
            bbox = detection['bbox']
            class_id = detection['class_id']

            # 为每个类别ID获取颜色
            color = get_color_for_class(class_id, color_map)

            # 在原始尺寸图像上绘制框
            x1 = int(bbox[0] * w_rate)
            y1 = int(bbox[1] * h_rate)
            x2 = int(bbox[2] * w_rate)
            y2 = int(bbox[3] * h_rate)

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)

            # 在同一行显示类别名称和置信度，带背景色
            label = f"{class_names[class_id]} {conf:.2f}"
            # 计算文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                  thickness)

            # 绘制文本背景框
            cv2.rectangle(result_image, (x1, y1 - text_height - baseline - 2),
                          (x1 + text_width, y1), color, -1)

            # 绘制文本
            cv2.putText(result_image, label, (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            print(f"检测框: 类别={class_names[class_id]}, 置信度={conf:.2f}")

    print(f"总共检测到 {detection_count} 个置信度高于阈值的目标")
    return result_image


def run_detection(model_path, image_path, class_names, conf_threshold=0.5):
    """
    运行目标检测

    Args:
        model_path (str): ONNX模型路径
        image_path (str): 图像路径
        conf_threshold (float): 置信度阈值
    """
    # 初始化模型
    model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # 读取图像
    image = cv2.imread(image_path)
    src_image = image.copy()

    # 预处理图像
    processed_image, scale_factors, original_size = preprocess_image(image)

    # 模型推理
    start = time.time()
    output = model.run(["dets", "labels"], {"input": processed_image})
    print(f"inference time: {time.time() - start}")
    bboxes = output[0][0]
    labels = output[1][0]

    # 后处理检测结果
    detections = postprocess_detections(bboxes, labels)

    # 绘制检测结果
    result_image = draw_detections(src_image, class_names, detections, scale_factors, conf_threshold)

    # 保存结果
    cv2.imwrite("result_onnx.jpg", result_image)


# 主程序
if __name__ == "__main__":
    model_path = "models/inference_model.onnx"
    image_path = "test/02_jpg.rf.65a084066fc353cd023eb5c953f40efe.jpg"
    start = time.time()
    class_names = ['ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan', 'human hauler',
                   'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter', 'suv', 'taxi',
                   'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow']
    run_detection(model_path, image_path,class_names, conf_threshold=0.5)
    print(f"total time: {time.time() - start}")
