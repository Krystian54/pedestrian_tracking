import cv2

# skrypt do odtwarzania plików wideo

cap = cv2.VideoCapture("data/wideo_1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Data", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
