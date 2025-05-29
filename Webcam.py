def webcam_detection():
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        frame = draw_labels(boxes, confs, colors, class_ids, classes, frame)
        
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

webcam_detection()