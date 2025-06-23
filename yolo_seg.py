import cv2
import numpy as np
from ultralytics import YOLO

# skrypt do dokonywania detekcji pieszych na nagraniu za pomocą sieci YOLO z segmentacją


model = YOLO("yolov8s-seg.pt")

cap = cv2.VideoCapture("data/wideo_1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # detekcja na pojedynczej klatce
    results = model(frame)

    # przejście po wykrytych obiektach
    for result in results:
        masks = result.masks  # segmentacje obiektów

        for box, mask in zip(result.boxes, masks.data if masks is not None else []):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            # wyrysowanie bounding boxa dla pieszych (klasa 0 dla COCO)
            if cls == 0 and conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)

                colored_mask = np.zeros_like(frame)
                colored_mask[:, :] = (0, 255, 0)
                mask_indices = mask[:, :, None]

                frame = np.where(mask_indices, cv2.addWeighted(frame, 1 - 0.5, colored_mask, 0.5, 0), frame)

    cv2.imshow("YOLO Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
