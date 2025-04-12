# Skrypt do dokonywania detekcji pieszych na nagraniu za pomocą sieci YOLO oraz ich identyfikacji z użyciem algorytmu węgierskiego


from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# współczynniki wagowe do obliczania kosztu
hog_ratio = 0.5   # współczynnik hog
dst_ratio = 0.1    # współczynnik odległości
speed_ratio = 0.4  # współczynnik prędkości
cost_treshold = 0.2   # próg na koszt

cap = cv2.VideoCapture("data/3647789-hd_1920_1080_30fps.mp4")
# cap = cv2.VideoCapture("data/854100-hd_1920_1080_25fps.mp4")

model = YOLO("yolov8n.pt")
hog = cv2.HOGDescriptor()

next_id = 0
tracked_objects = {}  # śledzone obiekty # id: {'hog': descriptor, 'bbox': (x1, y1, x2, y2), 'speed': speed}


# obliczenie hog
def compute_hog(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 128))
    descriptor = hog.compute(roi)
    return descriptor


# obliczenie środka ramki
def bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


# obliczenie kosztu
def combined_cost(hog1, hog2, bbox1, bbox2, speed1):
    hog_dist = np.linalg.norm(hog1 - hog2) / np.sqrt(3780) # normalizacja hog
    # print("hog", hog_dist)
    centroid1 = bbox_centroid(bbox1)
    centroid2 = bbox_centroid(bbox2)
    bbox_dist = np.linalg.norm(centroid1 - centroid2) / np.sqrt(1080**2 + 1920**2) # normalizacja dystansu
    # print("dst", bbox_dist)
    est_speed = centroid2 - centroid1  # estymowana prędkość
    speed_diff = np.linalg.norm(speed1 - est_speed) / np.sqrt(1080**2 + 1920**2) # normalizacja prędkości
    # print("speed", speed_diff)
    return hog_ratio * hog_dist + dst_ratio * bbox_dist + speed_ratio * speed_diff


# # zapis do wideo
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter('results/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # print(frame.shape)

    # YOLO
    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            if cls == 0 and conf > 0.5:
                detections.append((x1, y1, x2, y2))
                # rysowanie YOLO
                cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (0, 0, 255), 2)

    # obliczanie hog dla detekcji
    det_hogs = [compute_hog(frame, bbox) for bbox in detections]
    
    ids = list(tracked_objects.keys())  # aktualnie śledzone ID obiektów
    existing_data = [tracked_objects[tid] for tid in ids]   # dane dla aktualnych obiektów
    active_objects = {}  # tymczasowy słownik dla aktywnych obiektów w bieżącej klatce

    if existing_data and det_hogs:  # jeśli są zarówno śledzone obiekty, jak i nowe detekcje
        cost_matrix = np.zeros((len(existing_data), len(det_hogs)))  # macierz kosztów (rozmiar: śledzone x nowe)

        # wypełnianie macierzy kosztów
        for i, existing in enumerate(existing_data):
            for j, (d_hog, d_bbox) in enumerate(zip(det_hogs, detections)):
                # cost_matrix[i, j] = combined_cost(existing['hog'], d_hog, existing['bbox'], d_bbox)
                cost_matrix[i, j] = combined_cost(
                    existing['hog'], d_hog,
                    existing['bbox'], d_bbox,
                    existing['speed']
                )

        # algorytm węgierski
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_detections = set()  # set do przechowywania już dopasowanych wykryć

        # pętla po dopasowanych parach (istniejące ID, nowa detekcja)
        for r, c in zip(row_ind, col_ind):
            # print("cost", cost_matrix[r, c])
            if cost_matrix[r, c] < cost_treshold:
                tid = ids[r]  # pobranie istniejącego ID
                
                # aktualizacja obiektu
                # tracked_objects[tid] = {'hog': det_hogs[c], 'bbox': detections[c]}

                prev_centroid = bbox_centroid(tracked_objects[tid]['bbox'])
                new_centroid = bbox_centroid(detections[c])
                speed = new_centroid - prev_centroid

                tracked_objects[tid] = {
                    'hog': det_hogs[c],
                    'bbox': detections[c],
                    'speed': speed
                }

                active_objects[tid] = tracked_objects[tid]
                matched_detections.add(c)

        # tworzenie nowych obiektów dla niedopasowanych detekcji
        for i, (hog_vec, bbox) in enumerate(zip(det_hogs, detections)):
            if i not in matched_detections and hog_vec is not None:
                # tracked_objects[next_id] = {'hog': hog_vec, 'bbox': bbox}
                tracked_objects[next_id] = {
                    'hog': hog_vec,
                    'bbox': bbox,
                    'speed': np.array([0.0, 0.0])
                }
                active_objects[next_id] = tracked_objects[next_id]
                next_id += 1
    
    else:   # jeśli nie ma wcześniejszych obiektów lub detekcji — inicjalizacja wszystkich jako nowe
        for hog_vec, bbox in zip(det_hogs, detections):
            if hog_vec is not None:
                # tracked_objects[next_id] = {'hog': hog_vec, 'bbox': bbox}
                tracked_objects[next_id] = {
                    'hog': hog_vec,
                    'bbox': bbox,
                    'speed': np.array([0.0, 0.0])
                }
                active_objects[next_id] = tracked_objects[next_id]
                next_id += 1

    # rysowanie ramek i ID tylko dla dopasowanych lub nowych obiektów
    for tid, obj in active_objects.items():
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # out.write(frame)    # zapis

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# out.release()

cap.release()
cv2.destroyAllWindows()
