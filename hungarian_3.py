# Skrypt do dokonywania detekcji pieszych na nagraniu za pomocą sieci YOLO oraz ich identyfikacji z użyciem algorytmu węgierskiego + maska do HOG


from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# input_file = "data/kontener/Film3_cutted.mp4"
# output_file = "data/kontener/Film3_hungarian.mp4"

# input_file = "data/kontener/Film5_cutted.mp4"
# output_file = "data/kontener/Film5_hungarian.mp4"

input_file = "data/kontener/Film6_cutted.mp4"
output_file = "data/kontener/Film6_hungarian.mp4"


model = YOLO("yolov8s-seg.pt")
cap = cv2.VideoCapture(input_file)


hog_descriptor = cv2.HOGDescriptor()

# hog_descriptor = cv2.HOGDescriptor(
#     _winSize=(32, 64),
#     _blockSize=(8, 8),
#     _blockStride=(8, 8),
#     _cellSize=(8, 8),
#     _nbins=9
# )

# hog_descriptor = cv2.HOGDescriptor(
#     _winSize=(128, 256),
#     _blockSize=(16, 16),
#     _blockStride=(8, 8),
#     _cellSize=(8, 8),
#     _nbins=9
# )


# parametry
hog_ratio = 0.1 *           1    * (2+1)              #2
dst_ratio = 1/(1920/3) *    0.5  * (0.3*0)              #0.5
speed_ratio = 1/(1920/3) *  0.5  * (0.7*0)              #0.5
cost_threshold = np.inf                              #3
max_age = np.inf                                           #120

next_id = 0
tracked_objects = {}  # id: TrackedPerson

# klasa przechwująca parametry wykrytego obiektu
class TrackedPerson:
    def __init__(self, id, hog, bbox):
        self.id = id
        self.hog = hog
        self.bbox = bbox
        self.speed = np.array([0.0, 0.0])
        self.age = 0

    def update(self, hog, bbox):
        new_centroid = bbox_centroid(bbox)
        old_centroid = bbox_centroid(self.bbox)
        self.speed = new_centroid - old_centroid
        self.bbox = bbox
        self.hog = hog

    def update_age(self):
        self.age += 1

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
    # roi_resized = cv2.resize(roi_masked, (32, 64))
    # roi_resized = cv2.resize(roi_masked, (128, 256))

    result = hog_descriptor.compute(roi_resized)

    return result

# obliczenie kosztu
def combined_cost(hog1, hog2, bbox1, bbox2, speed1):
    hog_dist = np.linalg.norm(hog1 - hog2)
    centroid1 = bbox_centroid(bbox1)
    centroid2 = bbox_centroid(bbox2)
    bbox_dist = np.linalg.norm(centroid1 - centroid2)
    est_speed = centroid2 - centroid1
    speed_diff = np.linalg.norm(speed1 - est_speed)

    # print(hog_ratio * hog_dist + dst_ratio * bbox_dist + speed_ratio * speed_diff)
    return hog_ratio * hog_dist + dst_ratio * bbox_dist + speed_ratio * speed_diff

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

# zapis
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

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

    # # obliczenie HOG-a dla masek po erozji
    # det_hogs = [compute_hog_masked(frame, bbox, mask) for bbox, mask in zip(detections, eroded_masks)]



    # obliczenie hoga dla masek
    det_hogs = [compute_hog_masked(frame, bbox, mask) for bbox, mask in zip(detections, masks)]





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
            cost = combined_cost(
                active_objects[tid].hog,
                filtered_hogs[j],
                active_objects[tid].bbox,
                filtered_detections[j],
                active_objects[tid].speed
            )
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

out.release() # zapis

cap.release()
cv2.destroyAllWindows()
