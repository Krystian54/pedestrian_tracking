import cv2

# input_file = "data/kontener/Film3.mp4"
# output_file = "data/kontener/Film3_cutted.mp4"

# input_file = "data/kontener/Film5.mp4"
# output_file = "data/kontener/Film5_cutted.mp4"

# input_file = "data/kontener/Film6.mp4"
# output_file = "data/kontener/Film6_cutted.mp4"

start_time = 2 * 60 + 25
end_time = 3 * 60 + 25

cap = cv2.VideoCapture(input_file)


fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

current_frame = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or current_frame > end_frame:
        break

    if current_frame >= start_frame:
        out.write(frame)

    current_frame += 1

cap.release()
out.release()