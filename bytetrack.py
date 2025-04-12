# Skrypt do dokonywania detekcji pieszych na nagraniu za pomocą sieci YOLO oraz ich identyfikacji z użyciem bytetrack


from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data/3647789-hd_1920_1080_30fps.mp4")
# cap = cv2.VideoCapture("data/854100-hd_1920_1080_25fps.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # wykrywanie i śledzenie obiektów z użyciem DeepSORT
    results = model.track(frame, persist=True, tracker="./bytetrack.yaml", iou=0.2)
    
    # przejście po wykrytych obiektach
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else None
                
                # wyrysowanie bounding boxa dla pieszych (klasa 0 dla COCO)
                if cls == 0 and conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLOv8 + DeepSORT", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
