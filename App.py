import streamlit as st
import cv2
import uuid
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
from deep_sort_realtime.deepsort_tracker import DeepSort
import base64
import os

detector = MTCNN()
tracker = DeepSort(max_age=30, n_init=1)

def get_video_download_link(path, filename='processed-video.mp4'):
    with open(path, 'rb') as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    return f'<a href="data:file/mp4;base64,{b64}" download="{filename}">📥 Baixar vídeo processado</a>'

def run():
    st.title("🎥 Counting Unique Faces with Image Recognition")
    st.subheader("Upload a video (max 1 min)")

    video_file = st.file_uploader("Upload a video", type=['mp4', 'avi'])

    if video_file:
        input_path = f"input_{uuid.uuid4()}.mp4"
        with open(input_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = f"output_{uuid.uuid4()}.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        stframe = st.empty()
        progress_bar = st.progress(0)

        unique_ids = set()
        frame_id = 0
        interval = int(fps)

        st.info("⏳ Processing video with face identification...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_frame)
                detections = []

                for result in faces:
                    x, y, w, h = result['box']

                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = x1 + max(10, w)
                    y2 = y1 + max(10, h)

                    x2 = min(x2, rgb_frame.shape[1] - 1)
                    y2 = min(y2, rgb_frame.shape[0] - 1)

                    face_crop = rgb_frame[y1:y2, x1:x2]

                    if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                        continue

                    try:
                        rep = DeepFace.represent(face_crop, model_name='Facenet', enforce_detection=False)
                        emb = np.array(rep[0]['embedding'])
                    except:
                        continue

                    detections.append(([x1, y1, x2, y2], 0.99, "face", emb))

                tracks = tracker.update_tracks(detections, frame=rgb_frame)

                # Desenha só as caixas originais detectadas (uma por rosto)
                for bbox, _, _, _ in detections:
                    l, t, r, b = map(int, bbox)
                    cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)  # verde

                # Adiciona IDs rastreados no topo da caixa original
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    # Busca bbox da detecção atual do track (se disponível)
                    l, t, r, b = map(int, track.to_ltrb())
                    # Escreve ID próximo à caixa correspondente (aproximado)
                    cv2.putText(frame, f'ID {track_id}', (l, t - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    unique_ids.add(track_id)

                for _ in range(int(fps)):
                    out.write(frame)

                stframe.image(cv2.resize(frame, (640, 360)), channels="BGR")
                progress_bar.progress(min(int((frame_id / total_frames) * 100), 100))
            else:
                frame_id += 1
                continue

            frame_id += 1

        cap.release()
        out.release()

        st.success(f"✅ Processing done. Unique faces identified: {len(unique_ids)}")
        st.markdown(get_video_download_link(output_path), unsafe_allow_html=True)

        os.remove(input_path)

run()
