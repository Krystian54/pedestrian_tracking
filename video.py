# Skrypt do odtwarzania plików wideo


import cv2

cap = cv2.VideoCapture("data/3647789-hd_1920_1080_30fps.mp4")
# cap = cv2.VideoCapture("data/854100-hd_1920_1080_25fps.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Data", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
