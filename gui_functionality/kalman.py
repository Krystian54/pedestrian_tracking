import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ALgorytm węgierski z kalmanem


model = YOLO("yolov8n.pt")
hog = cv2.HOGDescriptor()


def compute_hog(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 128))
    descriptor = hog.compute(roi)
    return descriptor

def bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def init_kalman(centroid):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x[:2] = centroid.reshape(2, 1)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 100.
    kf.R *= 10.
    kf.Q = np.eye(4) * 1.
    return kf

def combined_cost(hog1, hog2, predicted_centroid, centroid2, speed1, hog_ratio_calc, dst_ratio_calc, speed_ratio_calc):
    hog_dist = np.linalg.norm(hog1 - hog2) / np.sqrt(3780)
    bbox_dist = np.linalg.norm(predicted_centroid - centroid2) / np.sqrt(1080**2 + 1920**2)
    est_speed = centroid2 - predicted_centroid
    speed_diff = np.linalg.norm(speed1 - est_speed) / np.sqrt(1080**2 + 1920**2)
    return hog_ratio_calc * hog_dist + dst_ratio_calc * bbox_dist + speed_ratio_calc * speed_diff

input_path = "data/3647789-hd_1920_1080_30fps.mp4"
output_path = 'kalman.avi'

def hungarian_algorithm_kalman(input_path, output_path, hog_ratio_gui, dst_ratio_gui, speed_ratio_gui, cost_threshold):
    next_id = 0 
    tracked_objects = {}  # id: {'hog': ..., 'bbox': ..., 'speed': ..., 'kf': KalmanFilter, 'age': ...}

    hog_ratio = hog_ratio_gui       #0.5
    dst_ratio = dst_ratio_gui       #0.2
    speed_ratio = speed_ratio_gui   #0.4

    cap = cv2.VideoCapture(input_path)
    # zapis
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # wyświetlenie obrazu w streamlit
    frame_slot = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                if cls == 0 and conf > 0.5:
                    detections.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (0, 0, 255), 2)

        det_hogs = [compute_hog(frame, bbox) for bbox in detections]

        ids = list(tracked_objects.keys())
        existing_data = [tracked_objects[tid] for tid in ids]
        active_objects = {}

        if existing_data and det_hogs:
            cost_matrix = np.zeros((len(existing_data), len(det_hogs)))

            for i, existing in enumerate(existing_data):
                predicted_centroid = existing['kf'].x[:2].ravel()
                for j, (d_hog, d_bbox) in enumerate(zip(det_hogs, detections)):
                    centroid2 = bbox_centroid(d_bbox)
                    cost_matrix[i, j] = combined_cost(
                        existing['hog'], d_hog,
                        predicted_centroid, centroid2,
                        existing['speed'],
                        hog_ratio_calc=hog_ratio,
                        dst_ratio_calc=dst_ratio,
                        speed_ratio_calc=speed_ratio)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_detections = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < cost_threshold:
                    tid = ids[r]
                    new_centroid = bbox_centroid(detections[c])
                    predicted_centroid = tracked_objects[tid]['kf'].x[:2].ravel()
                    speed = new_centroid - predicted_centroid

                    tracked_objects[tid]['hog'] = det_hogs[c]
                    tracked_objects[tid]['bbox'] = detections[c]
                    tracked_objects[tid]['speed'] = speed
                    tracked_objects[tid]['age'] = 0

                    tracked_objects[tid]['kf'].update(new_centroid)
                    active_objects[tid] = tracked_objects[tid]
                    matched_detections.add(c)

            for i, (hog_vec, bbox) in enumerate(zip(det_hogs, detections)):
                if i not in matched_detections and hog_vec is not None:
                    centroid = bbox_centroid(bbox)
                    kf = init_kalman(centroid)
                    tracked_objects[next_id] = {
                        'hog': hog_vec,
                        'bbox': bbox,
                        'speed': np.array([0.0, 0.0]),
                        'kf': kf,
                        'age': 0
                    }
                    active_objects[next_id] = tracked_objects[next_id]
                    next_id += 1

        else:
            for hog_vec, bbox in zip(det_hogs, detections):
                if hog_vec is not None:
                    centroid = bbox_centroid(bbox)
                    kf = init_kalman(centroid)
                    tracked_objects[next_id] = {
                        'hog': hog_vec,
                        'bbox': bbox,
                        'speed': np.array([0.0, 0.0]),
                        'kf': kf,
                        'age': 0
                    }
                    active_objects[next_id] = tracked_objects[next_id]
                    next_id += 1

        to_delete = []
        for tid in tracked_objects:
            if tid not in active_objects:
                tracked_objects[tid]['kf'].predict()
                tracked_objects[tid]['age'] += 1
                if tracked_objects[tid]['age'] < 300:
                    active_objects[tid] = tracked_objects[tid]
                else:
                    to_delete.append(tid)

        for tid in to_delete:
            del tracked_objects[tid]

        for tid, obj in active_objects.items():
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)  # zapisz klatkę do pliku

        # cv2.imshow("Result", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        frame_slot.image(frame, channels="BGR")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
