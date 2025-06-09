# Skrypt do dokonywania detekcji pieszych na nagraniu za pomocą sieci YOLO oraz ich identyfikacji z użyciem bytetrack
import streamlit as st

from ultralytics import YOLO
import cv2
import yaml


model = YOLO("yolov8n.pt")


def bytetrack_algorithm(input_path, output_path, iou):

    cap = cv2.VideoCapture(input_path)

    # zapis
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_slot = st.empty()  # miejsce na obraz w Streamlit

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # wykrywanie i śledzenie obiektów z użyciem DeepSORT
        results = model.track(frame, persist=True, tracker="./bytetrack.yaml", iou=iou)
        
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
        
        # cv2.imshow("YOLOv8 + DeepSORT", frame)
        frame_slot.image(frame, channels="BGR")  # wyświetlenie obrazu w Streamlit
        out.write(frame)  # zapis
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    out.release() # zapis

    cap.release()
    cv2.destroyAllWindows()

def set_bytetrack_parameters(
    path="./bytetrack.yaml",
    tracker_type="bytetrack",
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=300,
    match_thresh=0.9,
    fuse_score=True
):
    data = {
        "tracker_type": tracker_type,
        "track_high_thresh": track_high_thresh,
        "track_low_thresh": track_low_thresh,
        "new_track_thresh": new_track_thresh,
        "track_buffer": track_buffer,
        "match_thresh": match_thresh,
        "fuse_score": fuse_score
    }

    with open(path, "w") as plik:
        yaml.dump(data, plik, sort_keys=False)
