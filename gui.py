import os
import streamlit as st
from streamlit_option_menu import option_menu
from gui_functionality.bytetrack import bytetrack_algorithm, set_bytetrack_parameters
from gui_functionality.hungarian import hungarian_algorithm
from gui_functionality.kalman import hungarian_algorithm_kalman
from gui_functionality.hungarian_buffor import hungarian_algorithm_buffor


OUTPUT_NAME = 'gui_output.mp4'

st.set_page_config(page_title="Pedestrian tracking", layout="wide")
st.title("Pedestrian tracking")
st.write("Śledzenie osób w warunkach przesłaniania")

input_path = st.text_input("Ścieżka do filmu wejściowego", "data/wideo_kontener.mp4")
output_path = st.text_input("Katalog na film wyjściowy", "results")

st.subheader("Wybór algorytmu:")

# konfiguracja zakładek
selected_tab = option_menu(
    menu_title=None,
    options=["Algorytm bytetrack",
             "Algorytm węgierski (filtr kalmana)",
            "Algorytm węgierski (pozycja, prędkość, deskryptor HOG, rozmiar obiektu)",
            "Algorytm węgierski (pozycja, prędkość, deskryptor HOG, rozmiar obiektu, bufor)"],
    icons=["activity", "activity", "activity", "activity"],
    orientation="horizontal")

if selected_tab == "Algorytm bytetrack":
    st.subheader("Parametry algorytmu Bytetrack")
    tracker_type = st.selectbox("Typ trackera", ["bytetrack", "botsort"])
    iou = st.number_input("Próg obszaru IOU", min_value=0.0, max_value=1.0, value=0.2)
    track_high_thresh = st.number_input("Próg wysokiej pewności śledzenia", min_value=0.0, max_value=1.0, value=0.25)
    track_low_thresh = st.number_input("Próg niskiej pewności śledzenia", min_value=0.0, max_value=1.0, value=0.1)
    new_track_thresh = st.number_input("Próg tworzenia nowego śledzenia", min_value=0.0, max_value=1.0, value=0.25)
    track_buffer = st.number_input("Wielkość bufora", min_value=0, max_value=1000, value=300)
    match_thresh = st.number_input("Próg dopasowania", min_value=0.0, max_value=1.0, value=0.9)
    fuse_score = st.selectbox("Fuzja wyniku detekcji", [True, False])

elif selected_tab == "Algorytm węgierski (filtr kalmana)":
    st.subheader("Parametry algorytmu węgierskiego")
    hog_ratio = st.number_input("Współczynnik deskryptora HOG", min_value=0.0, value=0.5)
    dst_ratio = st.number_input("Współczynnik dystansu", min_value=0.0, value=0.2)
    speed_ratio = st.number_input("Współczynnik prędkości", min_value=0.0, value=0.4)
    cost_threshold = st.number_input("Próg na współczynniki", min_value=0.0, value=1.0)

elif selected_tab == "Algorytm węgierski (pozycja, prędkość, deskryptor HOG, rozmiar obiektu)":
    st.subheader("Parametry algorytmu węgierskiego")
    hog_ratio = st.number_input("Współczynnik deskryptora HOG", min_value=0.0, value=2.0)
    dst_ratio = st.number_input("Współczynnik dystansu", min_value=0.0, value=0.5)
    speed_ratio = st.number_input("Współczynnik prędkości", min_value=0.0, value=0.5)
    size_ratio = st.number_input("Współczynnik rozmiaru prostokąta otaczającego", min_value=0.0, value=1.0)
    cost_threshold = st.number_input("Próg na współczynniki", min_value=0.0, value=5.0)
    max_age = st.number_input("Maksymalny wiek obiektu", min_value=0.0, value=120.0)

elif selected_tab == "Algorytm węgierski (pozycja, prędkość, deskryptor HOG, rozmiar obiektu, bufor)":
    st.subheader("Parametry algorytmu węgierskiego")
    hog_ratio = st.number_input("Współczynnik deskryptora HOG", min_value=0.0, value=3.3)
    dst_ratio = st.number_input("Współczynnik dystansu", min_value=0.0, value=0.1)
    speed_ratio = st.number_input("Współczynnik prędkości", min_value=0.0, value=0.4)
    size_ratio = st.number_input("Współczynnik rozmiaru prostokąta otaczającego", min_value=0.0, value=0.2)
    cost_threshold = st.number_input("Próg na współczynniki", min_value=0.0, value=10.0)
    max_age = st.number_input("Maksymalny wiek obiektu", min_value=0.0, value=180.0)
    buffor_lenght = st.number_input("Długość bufora deskryptora HOG", min_value=1.0, value=10.0)
    buffor_type = st.selectbox("Typ bufora", ["średnia", "mediana"])

st.markdown("""
    <style>
    div.stButton > button {
        font-size: 20px;
        padding: 10px 50px;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    start = st.button("START")
with col2:
    stop = st.button("STOP")

if start ==  True:
    if not os.path.exists(input_path):
        st.error("Błąd: Ścieżka do pliku wejściowego nie istnieje.")
    if not os.path.isdir(output_path):
        st.error("Błąd: Ścieżka do katalogu wyjściowego nie istnieje.")

    output_path = os.path.join(output_path, OUTPUT_NAME)

    if selected_tab == "Algorytm bytetrack":
        set_bytetrack_parameters(
            path='/home/krystian/zzz_repozytoria/pedestrian_tracking/gui_functionality/bytetrack.yaml',
            tracker_type=tracker_type,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score)
        bytetrack_algorithm(input_path=input_path, output_path=output_path, iou=iou)
        st.success("Algorytm bytetrack zakończony sukcesem")

    elif selected_tab == "Algorytm węgierski (filtr kalmana)":
        hungarian_algorithm_kalman(
            input_path=input_path,
            output_path=output_path,
            hog_ratio_gui=hog_ratio,
            dst_ratio_gui=dst_ratio,
            speed_ratio_gui=speed_ratio,
            cost_threshold=cost_threshold)
        st.success("Algorytm węgierski zakończony sukcesem")

    elif selected_tab == "Algorytm węgierski (pozycja, prędkość, deskryptor HOG, rozmiar obiektu)":
        hungarian_algorithm(
            input_path=input_path,
            output_path=output_path,
            hog_ratio_gui=hog_ratio,
            dst_ratio_gui=dst_ratio,
            speed_ratio_gui=speed_ratio,
            size_ratio_gui=size_ratio,
            cost_threshold=cost_threshold,
            max_age=max_age)
        st.success("Algorytm węgierski zakończony sukcesem")

    elif selected_tab == "Algorytm węgierski (pozycja, prędkość, deskryptor HOG, rozmiar obiektu, bufor)":
        hungarian_algorithm_buffor(
            input_path=input_path,
            output_path=output_path,
            hog_ratio_gui=hog_ratio,
            dst_ratio_gui=dst_ratio,
            speed_ratio_gui=speed_ratio,
            size_ratio_gui=size_ratio,
            cost_threshold=cost_threshold,
            max_age=max_age,
            buffor_lenght=buffor_lenght,
            buffor_type=buffor_type)
        st.success("Algorytm węgierski zakończony sukcesem")

if stop == True:
    st.stop()

# do uruchomienia
# streamlit run gui.py
