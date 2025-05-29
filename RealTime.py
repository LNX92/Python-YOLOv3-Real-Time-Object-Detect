import cv2
import numpy as np
import time

# YOLO v3 Modulunu Yukleme
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

# Kare icerisinde bulunan objeleri yakalama
def detect_objects(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), 
                                mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outputs = net.forward(output_layers)
    end = time.time()
    return outputs, height, width, end-start

# Kutu Bilgilerini Getirme
def get_box_info(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
                
    return boxes, confs, class_ids

# Cizim ve Yazilari Getirme
def draw_boxes(frame, boxes, confs, colors, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)  # NMS threshold
    
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confs[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-5), font, 1, color, 2)
    return frame

# Gercek Zamanli Algilama
def realtime_detection():
    net, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)  # 0 Varsayilan kamerayi secer
    
    frame_count = 0
    total_fps = 0
    
    while True:
        _, frame = cap.read()
        if not _:
            break
            
        frame_count += 1
        start_time = time.time()
        
        # Islemi Algilama
        outputs, height, width, inference_time = detect_objects(frame, net, output_layers)
        boxes, confs, class_ids = get_box_info(outputs, height, width)
        frame = draw_boxes(frame, boxes, confs, colors, class_ids, classes)
        
        # FPS Hesaplama
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        total_fps += fps
        avg_fps = total_fps / frame_count
        
        # FPS Gosterme
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Real-time Object Detection", frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC Tusu Ile Cikis
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_detection()