# Interfejs graficzny zbierający wszystkie algorytmy


from gui_functionality.bytetrack import bytetrack_algorithm
from gui_functionality.bytetrack import set_bytetrack_parameters
from gui_functionality.hungarian_2 import hungarian_algorithm

import tkinter as tk
from tkinter import ttk, filedialog

OUTPUT_NAME = '/gui_output_video.mp4'

class Application:

    def __init__(self, root):

        self.root = root
        self.root.title("Śledzenie pieszych")
        self.root.geometry("630x330")

        self.input = tk.StringVar(value="/home/krystian/zzz_repozytoria/pedestrian_tracking/data/3647789-hd_1920_1080_30fps.mp4")
        self.output = tk.StringVar(value="/home/krystian/zzz_repozytoria/pedestrian_tracking/results")

        # parametry bytetrack
        self.bytetrack_iou = tk.StringVar(value = 0.2)
        self.bytetrack_tracker_type = tk.StringVar(value = "bytetrack")
        self.bytetrack_track_high_thresh = tk.StringVar(value = 0.25)
        self.bytetrack_track_low_thresh = tk.StringVar(value = 0.1)
        self.bytetrack_new_track_thresh = tk.StringVar(value = 0.25)
        self.bytetrack_track_buffer = tk.StringVar(value = 300)
        self.bytetrack_match_thresh = tk.StringVar(value = 0.9)
        self.bytetrack_fuse_score = tk.StringVar(value = True)

        # parametry hungarian
        self.hungarian_hog_ratio = tk.StringVar(value = 2)
        self.hungarian_dst_ratio = tk.StringVar(value = 0.5)
        self.hungarian_speed_ratio = tk.StringVar(value = 0.5)
        self.hungarian_cost_threshold = tk.StringVar(value = 3)
        self.hungarian_max_age = tk.StringVar(value = 120)

        self.create_top_panel()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=3, column=0, sticky="nsew")

        self.create_tabs()

    def create_top_panel(self):
        top_frame = tk.Frame(self.root)
        top_frame.grid(row=0, column=0, padx=5, pady=5)

        #input
        input_video_label = tk.Label(top_frame, text="Film wejściowy: ")
        input_video_label.grid(row=0, column=0, padx=5, pady=5)

        input_video_entry = tk.Entry(top_frame, textvariable=self.input, width=50)
        input_video_entry.grid(row=0, column=1, padx=5, pady=5)
        
        input_video_button = tk.Button(top_frame, text="Wybór", command=self.choose_input_file)
        input_video_button.grid(row=0, column=2, padx=5, pady=5)

        #output
        output_video_label = tk.Label(top_frame, text="Film wyjściowy: ")
        output_video_label.grid(row=1, column=0, padx=5, pady=5)

        output_video_entry = tk.Entry(top_frame, textvariable=self.output, width=50)
        output_video_entry.grid(row=1, column=1, padx=5, pady=5)
        
        output_video_button = tk.Button(top_frame, text="Wybór", command=self.choose_output_file)
        output_video_button.grid(row=1, column=2, padx=5, pady=5)

        # start
        start_button = tk.Button(top_frame, text="START", command=self.start_process)
        start_button.grid(row=2, column=2, padx=5, pady=5)

    def create_tabs(self):

        # zakładka 1
        frame_1 = tk.Frame(self.notebook)
        self.notebook.add(frame_1, text="Bytetrack")

        bytetrack_iou_label = tk.Label(frame_1, text="IOU: ")
        bytetrack_iou_label.grid(row=0, column=0, padx=5, pady=5)
        bytetrack_iou_entry = tk.Entry(frame_1, textvariable=self.bytetrack_iou, width=20)
        bytetrack_iou_entry.grid(row=0, column=1, padx=5, pady=5)

        bytetrack_tracker_type_label = tk.Label(frame_1, text="Tracker type ('botsort' or 'bytetrack'): ")
        bytetrack_tracker_type_label.grid(row=1, column=0)
        bytetrack_tracker_type_entry = tk.Entry(frame_1, textvariable=self.bytetrack_tracker_type, width=20)
        bytetrack_tracker_type_entry.grid(row=1, column=1)

        bytetrack_track_high_thresh_label = tk.Label(frame_1, text="Track high tresh: ")
        bytetrack_track_high_thresh_label.grid(row=2, column=0)
        bytetrack_track_high_thresh_entry = tk.Entry(frame_1, textvariable=self.bytetrack_track_high_thresh, width=20)
        bytetrack_track_high_thresh_entry.grid(row=2, column=1)

        bytetrack_track_low_thresh_label = tk.Label(frame_1, text="Track low tresh: ")
        bytetrack_track_low_thresh_label.grid(row=3, column=0)
        bytetrack_track_low_thresh_entry = tk.Entry(frame_1, textvariable=self.bytetrack_track_low_thresh, width=20)
        bytetrack_track_low_thresh_entry.grid(row=3, column=1)

        bytetrack_new_track_thresh_label = tk.Label(frame_1, text="New track tresh: ")
        bytetrack_new_track_thresh_label.grid(row=4, column=0)
        bytetrack_new_track_thresh_entry = tk.Entry(frame_1, textvariable=self.bytetrack_new_track_thresh, width=20)
        bytetrack_new_track_thresh_entry.grid(row=4, column=1)

        bytetrack_track_buffer_label = tk.Label(frame_1, text="Track buffer: ")
        bytetrack_track_buffer_label.grid(row=5, column=0)
        bytetrack_track_buffer_entry = tk.Entry(frame_1, textvariable=self.bytetrack_track_buffer, width=20)
        bytetrack_track_buffer_entry.grid(row=5, column=1)

        bytetrack_match_thresh_label = tk.Label(frame_1, text="Match tresh: ")
        bytetrack_match_thresh_label.grid(row=6, column=0)
        bytetrack_match_thresh_entry = tk.Entry(frame_1, textvariable=self.bytetrack_match_thresh, width=20)
        bytetrack_match_thresh_entry.grid(row=6, column=1)
        
        bytetrack_fuse_score_label = tk.Label(frame_1, text="Fuse score (True or False): ")
        bytetrack_fuse_score_label.grid(row=7, column=0)
        bytetrack_fuse_score_entry = tk.Entry(frame_1, textvariable=self.bytetrack_fuse_score, width=20)
        bytetrack_fuse_score_entry.grid(row=7, column=1)

        # zakładka 2
        frame_2 = tk.Frame(self.notebook)
        self.notebook.add(frame_2, text="Hungarian algorithm = HOG, distance, speed")

        hungarian_hog_ratio_label = tk.Label(frame_2, text="HOG ratio: ")
        hungarian_hog_ratio_label.grid(row=0, column=0, padx=5, pady=5)
        hungarian_hog_ratio_entry = tk.Entry(frame_2, textvariable=self.hungarian_hog_ratio, width=20)
        hungarian_hog_ratio_entry.grid(row=0, column=1, padx=5, pady=5)

        hungarian_dst_ratio_label = tk.Label(frame_2, text="DST ratio: ")
        hungarian_dst_ratio_label.grid(row=1, column=0)
        hungarian_dst_ratio_entry = tk.Entry(frame_2, textvariable=self.hungarian_dst_ratio, width=20)
        hungarian_dst_ratio_entry.grid(row=1, column=1)

        hungarian_speed_ratio_label = tk.Label(frame_2, text="SPEED ratio: ")
        hungarian_speed_ratio_label.grid(row=2, column=0)
        hungarian_speed_ratio_entry = tk.Entry(frame_2, textvariable=self.hungarian_speed_ratio, width=20)
        hungarian_speed_ratio_entry.grid(row=2, column=1)

        hungarian_cost_treshold_label = tk.Label(frame_2, text="COST threshold: ")
        hungarian_cost_treshold_label.grid(row=3, column=0)
        hungarian_cost_treshold_entry = tk.Entry(frame_2, textvariable=self.hungarian_cost_threshold, width=20)
        hungarian_cost_treshold_entry.grid(row=3, column=1)

        hungarian_max_age_label = tk.Label(frame_2, text="MAX AGE: ")
        hungarian_max_age_label.grid(row=4, column=0)
        hungarian_max_age_entry = tk.Entry(frame_2, textvariable=self.hungarian_max_age, width=20)
        hungarian_max_age_entry.grid(row=4, column=1)

    def choose_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.input.set(file_path)

    def choose_output_file(self):
        file_path = filedialog.askdirectory()
        if file_path:
            self.output.set(file_path)

    def start_process(self):

        input_path = self.input.get()
        output_path = self.output.get() + OUTPUT_NAME

        active_tab = self.notebook.index(self.notebook.select())

        if active_tab == 0:
            set_bytetrack_parameters(
                path='/home/krystian/zzz_repozytoria/pedestrian_tracking/gui_functionality/bytetrack.yaml',
                tracker_type=self.bytetrack_tracker_type.get(),
                track_high_thresh=float(self.bytetrack_track_high_thresh.get()),
                track_low_thresh=float(self.bytetrack_track_low_thresh.get()),
                new_track_thresh=float(self.bytetrack_new_track_thresh.get()),
                track_buffer=float(self.bytetrack_track_buffer.get()),
                match_thresh=float(self.bytetrack_match_thresh.get()),
                fuse_score=bool(self.bytetrack_fuse_score.get()))

            bytetrack_algorithm(input_path=input_path,
                                output_path=output_path,
                                iou=float(self.bytetrack_iou.get()))
        elif active_tab == 1:
            hungarian_algorithm(input_path=input_path,
                                output_path=output_path,
                                hog_ratio=float(self.hungarian_hog_ratio.get()),
                                dst_ratio=float(self.hungarian_dst_ratio.get()),
                                speed_ratio=float(self.hungarian_speed_ratio.get()),
                                cost_threshold=float(self.hungarian_cost_threshold.get()),
                                max_age=float(self.hungarian_max_age.get()))

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
