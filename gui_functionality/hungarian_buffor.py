import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# Algorytm węgierski: pozycja, prędkość, hog, wielkość ramki, bufor


model = YOLO("yolov8s-seg.pt")
hog_descriptor = cv2.HOGDescriptor()

# funkcja obsługi bufora
def buffor_append(list, new_hog):

    new_list = list

    if len(list) < BUFFOR_LENGHT:
        new_list.append(new_hog)
    else:
        del new_list[0]
        new_list.append(new_hog)

    return new_list

# klasa przechwująca parametry wykrytego obiektu
class TrackedPerson:
    def __init__(self, id, hog, bbox):
        self.id = id
        self.hog = hog
        self.bbox = bbox
        self.speed = np.array([0.0, 0.0])
        self.age = 0
        self.hog_bufor = [hog]

    def update(self, hog, bbox):
        new_centroid = bbox_centroid(bbox)
        old_centroid = bbox_centroid(self.bbox)
        self.speed = new_centroid - old_centroid
        self.bbox = bbox
        self.hog = hog
        self.hog_bufor = buffor_append(self.hog_bufor, self.hog)

    def update_age(self):
        self.age += 1

    def get_avg_hog(self):
        return np.mean(self.hog_bufor, axis=0)
    
    def get_med_hog(self):
        return np.median(self.hog_bufor, axis=0)

# obliczenie rozmiaru prostokąta otaczającego
def bbox_size(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([np.abs(x2 - x1), np.abs(y2 - y1)]) # szerokość i wysokość

# obliczenie środka prostokąta otaczającego
def bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

# obliczenie hog dla maski pieszego
def compute_hog_masked(frame, bbox, mask):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    mask_roi = mask[y1:y2, x1:x2]

    x, y, w, h = cv2.boundingRect(mask_roi)
    roi = roi[y:y+h, x:x+w]
    mask_roi = mask_roi[y:y+h, x:x+w]

    if roi.size == 0 or mask_roi.size == 0:
        return None

    roi_masked = cv2.bitwise_and(roi, roi, mask=mask_roi)
    roi_resized = cv2.resize(roi_masked, (64, 128))

    result = hog_descriptor.compute(roi_resized)

    return result

# obliczenie kosztu
def combined_cost(hog1, hog2, bbox1, bbox2, speed1, hog_ratio_calc, dst_ratio_calc, speed_ratio_calc, size_ratio_calc):
    hog_dist = np.linalg.norm(hog1 - hog2)
    centroid1 = bbox_centroid(bbox1)
    centroid2 = bbox_centroid(bbox2)
    bbox_dist = np.linalg.norm(centroid1 - centroid2)
    est_speed = centroid2 - centroid1
    speed_diff = np.linalg.norm(speed1 - est_speed)
    size1 = bbox_size(bbox1)
    size2 = bbox_size(bbox2)
    size_diff = (np.abs((size1[0]/size2[0])-1) + np.abs((size1[1]/size2[1])-1)) / 2

    return hog_ratio_calc * hog_dist + dst_ratio_calc * bbox_dist + speed_ratio_calc * speed_diff + size_ratio_calc * size_diff

# wyciagnięcie tylko pieszych z yolo razem z maskami
def process_detections(results, frame_shape):
    detections, masks = [], []

    for result in results:
        if result.boxes is None or result.masks is None:
            continue

        for box, mask_tensor in zip(result.boxes, result.masks.data):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if cls == 0 and conf > 0.5:
                detections.append((x1, y1, x2, y2))
                mask = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                masks.append(binary_mask)

    return detections, masks


def hungarian_algorithm_buffor(input_path, output_path, hog_ratio_gui, dst_ratio_gui, speed_ratio_gui, size_ratio_gui, cost_threshold, max_age, buffor_lenght, buffor_type):
  
    global BUFFOR_LENGHT
    BUFFOR_LENGHT = buffor_lenght

    cap = cv2.VideoCapture(input_path)

    # parametry
    hog_ratio = 0.1 *           1    * hog_ratio_gui                #3.3
    dst_ratio = 1/(1920/3) *    0.5  * dst_ratio_gui                #0.1
    speed_ratio = 1/(1920/3) *  0.5  * speed_ratio_gui              #0.4
    size_ratio = 10 *                   size_ratio_gui              #0.2


    next_id = 0
    tracked_objects = {}  # id: TrackedPerson

    # zapis
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # wyświetlenie obrazu w streamlit
    frame_slot = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # wyciągnięcie pieszych i masek
        detections, masks = process_detections(results, frame.shape)

        # Zdefiniuj jądro erozji
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))

        # Erozja masek
        eroded_masks = [cv2.erode(mask.astype(np.uint8), kernel, iterations=1) for mask in masks]

        masks = eroded_masks

        # obliczenie HOG-a dla masek po erozji
        det_hogs = [compute_hog_masked(frame, bbox, mask) for bbox, mask in zip(detections, eroded_masks)]

        # # obliczenie hoga dla masek
        # det_hogs = [compute_hog_masked(frame, bbox, mask) for bbox, mask in zip(detections, masks)]

        # filtracja hog
        valid_indices = [i for i, h in enumerate(det_hogs) if h is not None]
        filtered_detections = [detections[i] for i in valid_indices]
        filtered_hogs = [det_hogs[i] for i in valid_indices]
        filtered_masks = [masks[i] for i in valid_indices]

        # listy na aktywne obiekty
        active_objects = {k: v for k, v in tracked_objects.items() if v.age < max_age}
        active_ids = list(active_objects.keys())
        num_tracked = len(active_ids)
        num_detections = len(filtered_detections)

        # algorytm węgierski
        cost_matrix = np.full((num_tracked, num_detections), np.inf)

        for i, tid in enumerate(active_ids):
            for j in range(num_detections):

                if buffor_type == "średnia":
                    cost = combined_cost(
                        active_objects[tid].get_avg_hog(),
                        filtered_hogs[j],
                        active_objects[tid].bbox,
                        filtered_detections[j],
                        active_objects[tid].speed,
                        hog_ratio_calc=hog_ratio,
                        dst_ratio_calc=dst_ratio,
                        speed_ratio_calc=speed_ratio,
                        size_ratio_calc=size_ratio)
                elif buffor_type == "mediana":
                    cost = combined_cost(
                        active_objects[tid].get_med_hog(),
                        filtered_hogs[j],
                        active_objects[tid].bbox,
                        filtered_detections[j],
                        active_objects[tid].speed,
                        hog_ratio_calc=hog_ratio,
                        dst_ratio_calc=dst_ratio,
                        speed_ratio_calc=speed_ratio,
                        size_ratio_calc=size_ratio)
                    
                cost_matrix[i, j] = cost

        matched_rows, matched_cols = linear_sum_assignment(cost_matrix)

        matched_detections = set()
        used_ids = set()

        # aktualizacja obiektów
        for r, c in zip(matched_rows, matched_cols):
            if cost_matrix[r, c] < cost_threshold:
                tid = active_ids[r]
                active_objects[tid].update(filtered_hogs[c], filtered_detections[c])
                tracked_objects[tid] = active_objects[tid]
                matched_detections.add(c)
                used_ids.add(tid)

        # utworzenie nowych obiektów
        for i, (hog, bbox) in enumerate(zip(filtered_hogs, filtered_detections)):
            if i not in matched_detections:
                tracked_objects[next_id] = TrackedPerson(next_id, hog, bbox)
                next_id += 1

        # zwiększenie wieku obiektom "nie widzianym"
        for tid in tracked_objects:
            if tid not in used_ids:
                tracked_objects[tid].update_age()

        # filtracja aktywnych obiektów
        active_objects = {tid: tracked_objects[tid] for tid in used_ids}

        # wyrysowanie boundingboxów i masek
        for tid, obj in active_objects.items():
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # maska
        for tid, obj in active_objects.items():
            idx = list(active_objects.keys()).index(tid)
            if idx < len(masks):
                mask = masks[idx]
                colored_mask = np.zeros_like(frame)
                colored_mask[:, :] = (0, 255, 0)
                masked = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)
                frame = cv2.addWeighted(frame, 1.0, masked, 0.2, 0)

        out.write(frame)  # zapis

        # cv2.imshow("YOLO + HOG + Hungarian", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # wyświetlenie obrazu w streamlit
        frame_slot.image(frame, channels="BGR")
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    out.release() # zapis

    cap.release()
    cv2.destroyAllWindows()
