# ============================================================
# video_csv_camera_gui.py
# ------------------------------------------------------------
# 左   : USB カメラ (リアルタイム＋スナップショット)
# 中央 : アップロードした動画
# 右   : アップロードした CSV テーブル
# 列幅はサイドバーのスライダーで動的に変更可能
# ============================================================

import os
import cv2
import av
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading

# ─────────────────────────────────────────────────────────────
# Optional: Mediapipe pose overlay
# ─────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
except ModuleNotFoundError:
    mp = None  # pose overlay disabled


# ─────────────────────────────────────────────────────────────
# VideoProcessor for WebRTC (USB camera)
# ─────────────────────────────────────────────────────────────
class VideoProcessor(VideoProcessorBase):
    """Realtime processing pipeline:
       1. Optional horizontal live-crop
       2. Optional pose overlay
       3. ROI preview rectangle
    """
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None

        # crop parameters
        self.crop_enabled = False
        self.crop_width_percent = 100

        # pose overlay
        self.pose_enabled = False
        if mp is not None:
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.pose = None

        # ROI preview
        self.roi_enabled = False
        self.roi_x_start = 0
        self.roi_x_end   = 100
        self.roi_y_start = 0
        self.roi_y_end   = 100

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 1) horizontal crop
        if self.crop_enabled and 0 < self.crop_width_percent < 100:
            h, w, _ = img.shape
            new_w = int(w * self.crop_width_percent / 100)
            start = (w - new_w) // 2
            crop = img[:, start:start + new_w]
            letterbox = np.zeros_like(img)
            letterbox[:, start:start + new_w] = crop
            img = letterbox
        h, w = img.shape[:2]

        # 2) pose overlay
        if self.pose_enabled and self.pose is not None:
            res = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    res.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(thickness=2, circle_radius=2),
                )

        # 3) ROI rectangle
        if self.roi_enabled:
            x1 = int(w * self.roi_x_start / 100)
            x2 = int(w * self.roi_x_end   / 100)
            y1 = int(h * self.roi_y_start / 100)
            y2 = int(h * self.roi_y_end   / 100)
            x1, x2 = np.clip([x1, x2], 0, w - 1)
            y1, y2 = np.clip([y1, y2], 0, h - 1)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        with self.frame_lock:
            self.latest_frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─────────────────────────────────────────────────────────────
# Main Streamlit App
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Video / CSV / USB-Cam GUI", layout="wide")
st.title("🎬 Video・CSV・USB Camera Viewer (列幅調整可)")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    # ---------- Layout control ----------
    st.subheader("🖼️ レイアウト列幅")
    cam_ratio  = st.slider("左 (カメラ)", 1, 10, 3, help="st.columns の比率")
    vid_ratio  = st.slider("中央 (動画)", 1, 10, 4)
    csv_ratio  = st.slider("右 (CSV)",   1, 10, 3)

    # ---------- File inputs ----------
    st.subheader("📂 ファイルアップロード")
    video_file = st.file_uploader("📹 動画ファイル", type=["mp4", "mov", "avi", "mkv"])
    csv_file   = st.file_uploader("📑 CSV ファイル", type=["csv"])

    # ---------- Camera settings ----------
    st.subheader("⚙️ カメラ設定")
    save_dir = st.text_input("スナップショット保存先", "./snapshots")
    os.makedirs(save_dir, exist_ok=True)

    res_opts = {
        "640×480 (VGA)": (640, 480),
        "1280×720 (HD)": (1280, 720),
        "1920×1080 (Full HD)": (1920, 1080),
    }
    res_label = st.selectbox("解像度", list(res_opts.keys()), index=1)
    cam_w, cam_h = res_opts[res_label]

    if mp is None:
        st.warning("mediapipe 未インストール → `pip install mediapipe`")
        pose_toggle = False
    else:
        pose_toggle = st.checkbox("骨格描画", value=False)

    crop_toggle = st.checkbox("左右クロップ", value=False)
    crop_width_percent = st.slider("表示幅 (%)", 20, 100, 70, 5,
                                   disabled=not crop_toggle) if crop_toggle else 100

    st.markdown("**ROI (スナップショット範囲)**")
    roi_toggle = st.checkbox("ROI 指定", value=False)
    if roi_toggle:
        roi_x_start = st.slider("左 (%)", 0, 90, 0, 5)
        roi_x_end   = st.slider("右 (%)", roi_x_start + 10, 100, 100, 5)
        roi_y_start = st.slider("上 (%)", 0, 90, 0, 5)
        roi_y_end   = st.slider("下 (%)", roi_y_start + 10, 100, 100, 5)
    else:
        roi_x_start = roi_y_start = 0
        roi_x_end = roi_y_end = 100

# ============================================================
# 3-Column Layout (ratios are user-defined)
# ============================================================
col_cam, col_vid, col_csv = st.columns([cam_ratio, vid_ratio, csv_ratio], gap="medium")

# ----------------- USB Camera Column -----------------
with col_cam:
    st.subheader("📷 USB Camera")
    constraints = {
        "video": {
            "width":  {"ideal": cam_w},
            "height": {"ideal": cam_h},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    }
    webrtc_ctx = webrtc_streamer(
        key="usb_cam_stream",
        video_processor_factory=VideoProcessor,
        media_stream_constraints=constraints,
        async_processing=True,
    )

    # push state
    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        vp = webrtc_ctx.video_processor
        vp.pose_enabled = pose_toggle
        vp.crop_enabled = crop_toggle
        vp.crop_width_percent = crop_width_percent
        vp.roi_enabled = roi_toggle
        vp.roi_x_start, vp.roi_x_end = roi_x_start, roi_x_end
        vp.roi_y_start, vp.roi_y_end = roi_y_start, roi_y_end

    st.divider()
    c1, c2 = st.columns(2)
    snap_btn = c1.button("📸 SNAP", disabled=not webrtc_ctx.state.playing)
    del_btn  = c2.button("🗑️ 清掃", disabled=not os.listdir(save_dir))

    if snap_btn and webrtc_ctx.video_processor is not None:
        with webrtc_ctx.video_processor.frame_lock:
            frame = webrtc_ctx.video_processor.latest_frame
        if frame is None:
            st.warning("フレーム取得待ち ...")
        else:
            if roi_toggle:
                h, w = frame.shape[:2]
                x1 = int(w * roi_x_start / 100)
                x2 = int(w * roi_x_end   / 100)
                y1 = int(h * roi_y_start / 100)
                y2 = int(h * roi_y_end   / 100)
                frame = frame[y1:y2, x1:x2]
            fname = datetime.now().strftime("%Y%m%d_%H%M%S.png")
            path  = os.path.join(save_dir, fname)
            cv2.imwrite(path, frame)
            st.success(f"保存: {path}")

    if del_btn:
        removed = 0
        for f in os.listdir(save_dir):
            if f.lower().endswith(".png"):
                try:
                    os.remove(os.path.join(save_dir, f))
                    removed += 1
                except Exception as e:
                    st.error(f"{f} 削除失敗: {e}")
        st.warning(f"{removed} 枚削除")

# ----------------- Video Column -----------------
with col_vid:
    st.subheader("🎞️ 動画")
    if video_file is not None:
        st.video(video_file)
    else:
        st.info("動画ファイルをアップロードしてください。")

# ----------------- CSV Column -----------------
with col_csv:
    st.subheader("📑 CSV")
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"CSV 読込エラー: {e}")
    else:
        st.info("CSV ファイルをアップロードしてください.")

# ----------------- Hide default footer -----------------
st.markdown(
    "<style>footer {visibility: hidden;}</style>",
    unsafe_allow_html=True,
)
# ----------------- Hide default hamburger menu -----------------