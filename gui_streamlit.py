import streamlit as st
from streamlit_option_menu import option_menu
from gui_functionality.bytetrack import bytetrack_algorithm, set_bytetrack_parameters
from gui_functionality.hungarian_2 import hungarian_algorithm
import os

OUTPUT_NAME = 'gui_output_video.mp4'

def main():
    st.set_page_config(page_title="Śledzenie pieszych", layout="centered")

    st.title("Śledzenie pieszych")

    input_path = st.text_input("Ścieżka do filmu wejściowego", 
        "/home/krystian/zzz_repozytoria/pedestrian_tracking/data/3647789-hd_1920_1080_30fps.mp4")

    output_dir = st.text_input("Katalog na film wyjściowy", 
        "/home/krystian/zzz_repozytoria/pedestrian_tracking/results")

    selected_tab = option_menu(
        menu_title=None,
        options=["Bytetrack", "Algorytm węgierski"],
        icons=["activity", "activity"],
        orientation="horizontal"
    )

    if selected_tab == "Bytetrack":
        st.subheader("Parametry Bytetrack")

        tracker_type = st.selectbox("Tracker type", ["bytetrack", "botsort"])
        iou = st.number_input("IOU", min_value=0.0, max_value=1.0, value=0.2)
        track_high_thresh = st.number_input("Track high thresh", min_value=0.0, max_value=1.0, value=0.25)
        track_low_thresh = st.number_input("Track low thresh", min_value=0.0, max_value=1.0, value=0.1)
        new_track_thresh = st.number_input("New track thresh", min_value=0.0, max_value=1.0, value=0.25)
        track_buffer = st.number_input("Track buffer", min_value=0, max_value=1000, value=300)
        match_thresh = st.number_input("Match thresh", min_value=0.0, max_value=1.0, value=0.9)
        fuse_score = st.selectbox("Fuse score", [True, False])

    elif selected_tab == "Algorytm węgierski":
        st.subheader("Parametry algorytmu węgierskiego")

        hog_ratio = st.number_input("HOG ratio", min_value=0.0, value=2.0)
        dst_ratio = st.number_input("DST ratio", min_value=0.0, value=0.5)
        speed_ratio = st.number_input("Speed ratio", min_value=0.0, value=0.5)
        cost_threshold = st.number_input("Cost threshold", min_value=0.0, value=3.0)
        max_age = st.number_input("Max age", min_value=0.0, value=120.0)

    if st.button("START"):
        if not os.path.exists(input_path):
            st.error("Błąd: Ścieżka do pliku wejściowego nie istnieje.")
            return
        if not os.path.isdir(output_dir):
            st.error("Błąd: Ścieżka do katalogu wyjściowego nie istnieje.")
            return

        output_path = os.path.join(output_dir, OUTPUT_NAME)

        if selected_tab == "Bytetrack":
            set_bytetrack_parameters(
                path='/home/krystian/zzz_repozytoria/pedestrian_tracking/gui_functionality/bytetrack.yaml',
                tracker_type=tracker_type,
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                fuse_score=fuse_score
            )
            bytetrack_algorithm(input_path=input_path, output_path=output_path, iou=iou)
            st.success("Bytetrack success")

        elif selected_tab == "Hungarian":
            hungarian_algorithm(
                input_path=input_path,
                output_path=output_path,
                hog_ratio=hog_ratio,
                dst_ratio=dst_ratio,
                speed_ratio=speed_ratio,
                cost_threshold=cost_threshold,
                max_age=max_age
            )
            st.success("Hungarian success")


if __name__ == "__main__":
    main()


# do uruchomienia
# streamlit run gui_streamlit.py
