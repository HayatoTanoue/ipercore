# streamlit_triple_view.py  (v4.2 – better status & error handling)
"""
* Adds clear status messages:
  * "⏳ Waiting for frames…" while camera starts.
  * Error if the device cannot be opened.
* Resets `current_frame` and any previous error on start.
* Worker stores `error_msg` in `st.session_state` if capture fails.
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except ImportError:
    st_autorefresh = None  # type: ignore

###############################################################################
# --------------------------- Helper functions ------------------------------ #
###############################################################################

def _camera_worker(cam_index: int, width: int, height: int, compress: bool) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        st.session_state.error_msg = f"⚠️ Could not open /dev/video{cam_index}."
        st.session_state.camera_on = False
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    st.session_state.pop("error_msg", None)  # clear any previous error
    while st.session_state.get("camera_on", False):
        ok, frame = cap.read()
        if not ok:
            continue
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if compress:
            _, jpg = cv2.imencode(".jpg", frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            st.session_state.current_frame = jpg.tobytes()
            st.session_state.frame_format = "jpg"
        else:
            st.session_state.current_frame = frame_rgb
            st.session_state.frame_format = "raw"
    cap.release()
    st.session_state.pop("camera_thread", None)


def embed_video(path: Path, loop: bool) -> None:
    mime = "video/mp4"
    data_uri = base64.b64encode(path.read_bytes()).decode()
    loop_attr = "loop" if loop else ""
    st.markdown(
        f"""
        <video controls {loop_attr} style='width:100%;height:auto;'>
            <source src='data:{mime};base64,{data_uri}' type='{mime}'>
        </video>
        """,
        unsafe_allow_html=True,
    )

###############################################################################
# ----------------------------------  Main ---------------------------------- #
###############################################################################

def main() -> None:
    st.set_page_config(page_title="Triple-View Dashboard", layout="wide")
    st.title("📹 Triple-View Streamlit Dashboard – v4.2")

    st.session_state.setdefault("camera_on", False)
    st.session_state.setdefault("current_frame", None)
    st.session_state.setdefault("frame_format", "raw")

    col_cam, col_vid, col_csv = st.columns(3)

    # --------------------------- Camera Pane --------------------------- #
    with col_cam:
        st.subheader("USB Camera")
        cam_index = st.number_input("Camera index", 0, 10, 0, 1)
        target_fps = st.slider("Target FPS", 1, 60, 15)
        downscale = st.checkbox("Downscale to 320×240", True)
        compress = st.checkbox("JPEG compress", True)

        if st.button("Start ❯"):
            st.session_state.camera_on = True
            st.session_state.current_frame = None  # reset
            st.session_state.pop("error_msg", None)
        if st.button("Stop ■"):
            st.session_state.camera_on = False

        w, h = (320, 240) if downscale else (640, 480)
        if st.session_state.camera_on and "camera_thread" not in st.session_state:
            th = threading.Thread(
                target=_camera_worker,
                args=(int(cam_index), w, h, compress),
                daemon=True,
            )
            th.start()
            st.session_state.camera_thread = th

        frame_ph = st.empty()
        if "error_msg" in st.session_state:
            frame_ph.error(st.session_state.error_msg)
        elif st.session_state.current_frame is not None:
            if st.session_state.frame_format == "jpg":
                frame_ph.image(
                    st.session_state.current_frame, output_format="JPEG", channels="RGB"
                )
            else:
                frame_ph.image(st.session_state.current_frame, channels="RGB")
        elif st.session_state.camera_on:
            frame_ph.info("⏳ Waiting for frames…")
        else:
            frame_ph.info("Camera is off.")

    # ---------------- Video Pane ---------------- #
    with col_vid:
        st.subheader("Video Player")
        p = st.text_input("Video file path (MP4)")
        if p:
            fp = Path(p).expanduser()
            if fp.is_file():
                embed_video(fp, st.checkbox("Loop", True))
            else:
                st.error(f"File not found: {fp}")
        else:
            st.info("Enter a path to a video file.")

    # ---------------- CSV Pane ---------------- #
    with col_csv:
        st.subheader("CSV Viewer")
        cp_str = st.text_input("CSV file path")
        if cp_str:
            cp = Path(cp_str).expanduser()
            if cp.is_file():
                try:
                    df = pd.read_csv(cp)
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
            else:
                st.error(f"File not found: {cp}")
        else:
            st.info("Enter a path to a CSV file.")

    # -------------- Refresh logic -------------- #
    if st.session_state.camera_on:
        interval_ms = int(1000 / target_fps)
        if hasattr(st, "experimental_rerun"):
            time.sleep(interval_ms / 1000)
            st.experimental_rerun()
        elif st_autorefresh is not None:
            st_autorefresh(interval=interval_ms, key="cam_refresh")


if __name__ == "__main__":
    main()
