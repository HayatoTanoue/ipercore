import os
import cv2
import av
import numpy as np
import streamlit as st
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading

# Pose estimation (MediaPipe)
try:
    import mediapipe as mp
except ModuleNotFoundError:
    mp = None  # Pose overlay disabled if not installed


# -----------------------------
# Utility
# -----------------------------

def hex_to_bgr(hex_color: str):
    """Convert #RRGGBB to BGR tuple (for OpenCV/MediaPipe)"""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# -----------------------------
# VideoProcessor for WebRTC (USB camera)
# -----------------------------
class VideoProcessor(VideoProcessorBase):
    """Realtime processing pipeline with customizable pose skeleton style."""

    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_frame = None  # numpy.ndarray (BGR)

        # Live crop controls
        self.crop_enabled = False
        self.crop_width_percent = 100

        # Pose overlay & style
        self.pose_enabled = False
        self.skeleton_color_bgr = (0, 255, 0)
        self.skeleton_thickness = 2
        self.skeleton_circle_radius = 2

        # Snapshot ROI preview controls
        self.roi_enabled = False
        self.roi_x_start = 0
        self.roi_x_end = 100
        self.roi_y_start = 0
        self.roi_y_end = 100

        # MediaPipe Pose setup
        if mp is not None:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(static_image_mode=False,
                                           model_complexity=1,
                                           enable_segmentation=False,
                                           min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
        else:
            self.pose = None

    # -------------------------
    # Main video callback
    # -------------------------
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 1. Horizontal live crop (letter‑box)
        if self.crop_enabled and 0 < self.crop_width_percent < 100:
            h, w_orig, _ = img.shape
            new_w = int(w_orig * self.crop_width_percent / 100)
            start_x = (w_orig - new_w) // 2
            cropped = img[:, start_x:start_x + new_w]
            img_letterbox = np.zeros_like(img)
            img_letterbox[:, start_x:start_x + new_w] = cropped
            img = img_letterbox
        h, w = img.shape[:2]

        # 2. Pose overlay
        if self.pose_enabled and self.pose is not None:
            results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                draw_spec = self.mp_drawing.DrawingSpec(color=self.skeleton_color_bgr,
                                                         thickness=self.skeleton_thickness,
                                                         circle_radius=self.skeleton_circle_radius)
                self.mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=draw_spec,
                    connection_drawing_spec=draw_spec,
                )

        # 3. ROI preview rectangle
        if self.roi_enabled:
            x1 = int(w * self.roi_x_start / 100)
            x2 = int(w * self.roi_x_end / 100)
            y1 = int(h * self.roi_y_start / 100)
            y2 = int(h * self.roi_y_end / 100)
            x1, x2 = np.clip([x1, x2], 0, w - 1)
            y1, y2 = np.clip([y1, y2], 0, h - 1)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        with self.frame_lock:
            self.latest_frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Realtime USB Camera Viewer", layout="wide")
    st.title("📷 USB Camera Realtime Viewer & Snapshot Tool")

    # Sidebar
    st.sidebar.header("Settings")
    save_dir = st.sidebar.text_input("Image save directory", "./snapshots")
    os.makedirs(save_dir, exist_ok=True)

    # Resolution
    res_options = {"640×480 (VGA)": (640, 480), "1280×720 (HD)": (1280, 720), "1920×1080 (Full HD)": (1920, 1080)}
    res_label = st.sidebar.selectbox("Camera resolution", list(res_options.keys()), index=1)
    cam_w, cam_h = res_options[res_label]

    # Pose overlay & style
    pose_toggle = False
    if mp is None:
        st.sidebar.warning("mediapipe 未インストール。`pip install mediapipe` で骨格描画を有効化できます。")
    else:
        pose_toggle = st.sidebar.checkbox("Show pose skeleton overlay", value=False)
        if pose_toggle:
            col1, col2 = st.sidebar.columns(2)
            skel_color_hex = col1.color_picker("Skeleton color", "#00FF00")
            skel_thick = col2.slider("Thickness", 1, 10, 2)
            skel_radius = st.sidebar.slider("Landmark radius", 1, 10, 2)
        else:
            skel_color_hex = "#00FF00"
            skel_thick = 2
            skel_radius = 2

    # Live crop
    crop_toggle = st.sidebar.checkbox("Live crop horizontally", value=False)
    crop_width_percent = st.sidebar.slider("Visible width (%)", 20, 100, 70, 5) if crop_toggle else 100

    # Snapshot ROI
    st.sidebar.markdown("---")
    st.sidebar.subheader("Snapshot ROI preview")
    roi_toggle = st.sidebar.checkbox("Display & save ROI", value=False)
    if roi_toggle:
        roi_x_start = st.sidebar.slider("Left (%)", 0, 90, 0, 5)
        roi_x_end   = st.sidebar.slider("Right (%)", roi_x_start + 10, 100, 100, 5)
        roi_y_start = st.sidebar.slider("Top (%)", 0, 90, 0, 5)
        roi_y_end   = st.sidebar.slider("Bottom (%)", roi_y_start + 10, 100, 100, 5)
    else:
        roi_x_start = roi_y_start = 0
        roi_x_end = roi_y_end = 100

    st.sidebar.markdown("---")
    st.sidebar.write("Snapshots are saved as PNG with timestamps.")

    # WebRTC stream
    constraints = {"video": {"width": {"ideal": cam_w}, "height": {"ideal": cam_h}, "frameRate": {"ideal": 30}}, "audio": False}
    webrtc_ctx = webrtc_streamer(key=f"usb_cam_stream_{cam_w}x{cam_h}", video_processor_factory=VideoProcessor, media_stream_constraints=constraints, async_processing=True)

    # Pass states to processor
    if webrtc_ctx.state.playing and webrtc_ctx.video_processor is not None:
        vp = webrtc_ctx.video_processor
        vp.pose_enabled = pose_toggle
        vp.skeleton_color_bgr = hex_to_bgr(skel_color_hex)
        vp.skeleton_thickness = skel_thick
        vp.skeleton_circle_radius = skel_radius
        vp.crop_enabled = crop_toggle
        vp.crop_width_percent = crop_width_percent
        vp.roi_enabled = roi_toggle
        vp.roi_x_start, vp.roi_x_end = roi_x_start, roi_x_end
        vp.roi_y_start, vp.roi_y_end = roi_y_start, roi_y_end

    # Buttons
    col1, col2 = st.columns(2)
    snap_btn = col1.button("📸 Capture snapshot")
    del_btn = col2.button("🗑️ Delete all images")

    # Snapshot logic
    if snap_btn and webrtc_ctx.video_processor is not None:
        with webrtc_ctx.video_processor.frame_lock:
            frame = webrtc_ctx.video_processor.latest_frame
        if frame is None:
            st.warning("No frame available to capture.")
        else:
            if roi_toggle:
                h, w = frame.shape[:2]
                x1 = int(w * roi_x_start / 100)
                x2 = int(w * roi_x_end / 100)
                y1 = int(h * roi_y_start / 100)
                y2 = int(h * roi_y_end / 100)
                frame = frame[y1:y2, x1:x2]
            filename = datetime.now().strftime("%Y%m%d_%H%M%S.png")
            cv2.imwrite(os.path.join(save_dir, filename), frame)
            st.success(f"Saved snapshot → {filename}")

    # Delete logic
    if del_btn:
        removed = sum(1 for f in os.listdir(save_dir) if f.lower().endswith(".png"))
        for f in list(os.listdir(save_dir)):
            if f.lower().endswith(".png"):
                try:
                    os.remove(os.path.join(save_dir, f))
                except Exception as e:
                    st.error(f"Failed to delete {f}: {e}")
        st.warning(f"Deleted {removed} PNG file(s)")

    st.markdown(
        """
        **Usage tips**  
        • Adjust skeleton color, thickness, and landmark radius when pose overlay is on.  
        • Green rectangle shows the exact ROI that will be saved.  
        • Live horizontal crop narrows the FOV with black bars.  
        • All settings apply instantly, no reload needed.
        """
    )


if __name__ == "__main__":
    main()
