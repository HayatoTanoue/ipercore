# streamlit_triple_view.py
"""
Streamlit GUI with three side-by-side panes:
1. Live USB camera preview (OpenCV).
2. Video player for an arbitrary file path (supports optional loop\, seek via native controls).
3. CSV loader with interactive table display.

Run with:
    streamlit run streamlit_triple_view.py

Dependencies:
    pip install streamlit opencv-python pandas

Notes:
* The camera reader runs in a background thread so the Streamlit event loop stays responsive.
* Large video files are served via an <video> tag encoded as base64 to enable looping.  For local
  networks this avoids CORS issues.  For very large files you may prefer hosting the video and
  simply pointing the <video> tag to its URL instead of embedding.
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

###############################################################################
# ---------------------------- Helper functions ----------------------------- #
###############################################################################

def _camera_loop(placeholder: st.delta_generator.DeltaGenerator, cam_index: int) -> None:
    """Continuously grab frames from the specified camera index and display them."""
    cap = cv2.VideoCapture(cam_index)
    # Try to set a reasonable resolution; ignore failures quietly.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Stream until the flag is flipped off or the capture fails.
    while st.session_state.get("camera_on", False):
        ret, frame = cap.read()
        if not ret:
            placeholder.error("⚠️  Failed to read from camera.")
            break
        # Convert BGR ➜ RGB for correct color rendering in Streamlit.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(frame_rgb, channels="RGB")
        # Throttle a bit so we do not saturate the CPU.
        time.sleep(0.03)

    cap.release()
    placeholder.info("Camera stopped.")


def _embed_video(path: Path, loop: bool) -> None:
    """Render a local video file using an HTML <video> tag with optional looping."""
    mime = "video/mp4"  # You can extend this if you need other formats.
    video_bytes = path.read_bytes()
    data_uri = base64.b64encode(video_bytes).decode("utf-8")
    loop_attr = "loop" if loop else ""
    video_html = f"""
    <video controls {loop_attr} style="width: 100%; height: auto;">
        <source src="data:{mime};base64,{data_uri}" type="{mime}">
        Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

###############################################################################
# ---------------------------------  Main  ---------------------------------- #
###############################################################################

def main() -> None:
    st.set_page_config(
        page_title="Triple-View Dashboard",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.title("📹 Triple-View Streamlit Dashboard")
    st.markdown(
        "Preview your **USB camera**, play a **local video file**, and inspect a **CSV** all at once."
    )

    # Three equal-width columns.
    col_camera, col_video, col_csv = st.columns(3)

    # ------------------------------------------------------------------
    # 🟢 1) USB camera pane
    # ------------------------------------------------------------------
    with col_camera:
        st.subheader("USB Camera")
        cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
        placeholder = st.empty()

        # Session flag keeps state across reruns.
        if "camera_on" not in st.session_state:
            st.session_state["camera_on"] = False

        if st.button("Start Camera" if not st.session_state["camera_on"] else "Stop Camera"):
            st.session_state["camera_on"] = not st.session_state["camera_on"]
            if st.session_state["camera_on"]:
                # Launch background reader.
                threading.Thread(
                    target=_camera_loop, args=(placeholder, int(cam_index)), daemon=True
                ).start()

        # Initial placeholder content.
        if not st.session_state["camera_on"]:
            placeholder.info("Camera is off.")

    # ------------------------------------------------------------------
    # 🔵 2) Video file pane
    # ------------------------------------------------------------------
    with col_video:
        st.subheader("Video Player")
        video_path_str = st.text_input("Video file path (MP4 recommended)")
        loop_playback = st.checkbox("Loop playback", value=True, help="Replay automatically when finished")

        if video_path_str:
            video_path = Path(video_path_str).expanduser()
            if video_path.is_file():
                _embed_video(video_path, loop_playback)
            else:
                st.error(f"File not found: {video_path}")
        else:
            st.info("Enter a path to a video file above.")

    # ------------------------------------------------------------------
    # 🟣 3) CSV table pane
    # ------------------------------------------------------------------
    with col_csv:
        st.subheader("CSV Viewer")
        csv_path_str = st.text_input("CSV file path")
        if csv_path_str:
            csv_path = Path(csv_path_str).expanduser()
            if csv_path.is_file():
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
                else:
                    st.dataframe(df, use_container_width=True)
            else:
                st.error(f"File not found: {csv_path}")
        else:
            st.info("Enter a path to a CSV file above.")


if __name__ == "__main__":
    main()
