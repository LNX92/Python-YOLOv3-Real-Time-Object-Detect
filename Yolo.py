import cv2
import numpy as np

# Load YOLOv3 model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

# Detect objects in image
def detect_objects(img, net, outputLayers):            
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

# Get bounding boxes and confidence scores
def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:  # Confidence threshold
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
                
    return boxes, confs, class_ids

# Draw bounding boxes on image
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)  # Non-max suppression
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    return img

# Main function
def image_detection(image_path): 
    model, classes, colors, output_layers = load_yolo()
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    
    blob, outputs = detect_objects(img, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    img = draw_labels(boxes, confs, colors, class_ids, classes, img)
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "C:/Users/Oto_Test2/Desktop/YOLO_V3/Project_6_YOLO/dog_human.jpg"
image_detection(image_path)