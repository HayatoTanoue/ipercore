# -*- coding: utf-8 -*-
"""
Streamlit GUI for Human Motion Imitation / Novel View Synthesis / Human Appearance Transfer
=========================================================================================
This GUI wraps the command‑line tools described in the README and provides:
    • Task selection (motion_imitate.py, novel_view.py, appearance_transfer.py)
    • Widget‑driven parameter input (gpu_ids, image_size, model_id, etc.)
    • Automatic **num_source** detection from a source directory
    • Builder helpers to generate *src_path* / *ref_path* strings
    • Two‑step, CPU‑only, fast video synthesis (PNG → FFV1 → H.264)
    • Execution logs and in‑app video preview
The video creation part now uses a two‑stage pipeline that is 2‑5× faster than
single‑stage libx264 when only CPUs are available.
"""

import os
import glob
import subprocess
import time
import pathlib
from datetime import datetime

import streamlit as st

##########################################################################################
# Helper utilities
##########################################################################################

def count_images_in_dir(directory: str) -> int:
    """Return the number of image files (common extensions) in *directory*."""
    patterns = [
        ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp",
        ".PNG", ".JPG", ".JPEG", ".BMP", ".GIF", ".WEBP",
    ]
    files = [p for p in glob.glob(os.path.join(directory, "*")) if os.path.splitext(p)[1] in patterns]
    return len(files)

##########################################################################################
# Streamlit Page config & Sidebar – global parameters
##########################################################################################

st.set_page_config(page_title="Human Motion GUI", layout="wide")

st.title("Human Motion / Appearance Transfer GUI")

with st.sidebar:
    st.header("Global Parameters")
    gpu_ids = st.text_input("gpu_ids", "0")
    image_size = st.number_input("image_size", min_value=128, max_value=2048, value=512, step=64)
    assets_dir = st.text_input("assets_dir", "./assets")
    output_dir = st.text_input("output_dir", "./results")
    model_id = st.text_input("model_id (leave blank → auto)", "")
    st.markdown("---")

##########################################################################################
# Task selection
##########################################################################################

mode = st.radio("Select Task", ["Motion Imitation", "Novel View", "Appearance Transfer"], horizontal=True)

SCRIPT_MAP = {
    "Motion Imitation": "demo/motion_imitate.py",
    "Novel View": "demo/novel_view.py",
    "Appearance Transfer": "demo/appearance_transfer.py",
}

##########################################################################################
# Quick Builder – minimal inputs that cover most use‑cases
##########################################################################################

st.subheader("Quick Builder")
quick_col1, quick_col2 = st.columns(2)

with quick_col1:
    src_dir = st.text_input("Source directory / image / video", key="src_dir")
    if src_dir and os.path.exists(src_dir):
        img_cnt = count_images_in_dir(src_dir) or 1
        st.info(f"Detected **{img_cnt}** image(s) under source directory.")
    else:
        img_cnt = 2

with quick_col2:
    ref_path_simple = st.text_input("Reference path (video / image / dir)", key="ref_simple")

num_source = st.number_input(
    "num_source (auto‑filled from src_dir)",
    min_value=1,
    max_value=20,
    value=img_cnt,
    step=1,
)

# Optional reference parameters
with st.expander("Optional reference parameters"):
    ref_name = st.text_input("name? (ref)")
    ref_fps = st.number_input("fps?", min_value=1, max_value=120, value=25, step=1)
    pose_fc = st.number_input("pose_fc?", min_value=10, max_value=1000, value=300, step=10)
    cam_fc = st.number_input("cam_fc?", min_value=10, max_value=1000, value=150, step=10)
    effect = st.text_input("effect? (e.g. View‑45;BT‑30‑180)")

##########################################################################################
# Build src_path and ref_path (unless overridden)
##########################################################################################

def build_src_path() -> str:
    name = pathlib.Path(src_dir).stem
    return f"path?={src_dir},name?={name}"


def build_ref_path() -> str:
    fields = [f"path?={ref_path_simple}"]
    if ref_name:
        fields.append(f"name?={ref_name}")
    if ref_fps:
        fields.append(f"fps?={int(ref_fps)}")
    if pose_fc:
        fields.append(f"pose_fc?={int(pose_fc)}")
    if cam_fc:
        fields.append(f"cam_fc?={int(cam_fc)}")
    if effect:
        fields.append(f"effect?={effect}")
    return ",".join(fields)

##########################################################################################
# Novel‑View‑specific flag
##########################################################################################

T_POSE_FLAG = ""
if mode == "Novel View":
    render_tpose = st.checkbox("Render T‑pose", value=False)
    if render_tpose:
        T_POSE_FLAG = " --T_pose"

##########################################################################################
# Run button
##########################################################################################

run_btn = st.button("🔥 Run Script 🔥")

if run_btn:
    src_path_final = build_src_path()
    ref_path_final = build_ref_path()

    # Basic validation
    if not src_path_final:
        st.error("src_path is required (provide via Quick Builder or Advanced field).")
        st.stop()

    if mode != "Novel View" and not ref_path_final:
        st.error("ref_path is required for this task.")
        st.stop()

    # Auto model_id if empty
    _model_id = model_id or (pathlib.Path(src_dir).stem if src_dir else "custom_model")

    cmd = (
        f"python {SCRIPT_MAP[mode]} "
        f"--gpu_ids {gpu_ids} "
        f"--image_size {image_size} "
        f"--num_source {num_source} "
        f"--output_dir \"{output_dir}\" "
        f"--assets_dir \"{assets_dir}\" "
        f"--model_id \"{_model_id}\" "
        f"--src_path \"{src_path_final}\""
    )

    if mode != "Novel View":
        cmd += f" --ref_path \"{ref_path_final}\""
    cmd += T_POSE_FLAG

    st.code(cmd, language="bash")

    with st.spinner("Executing… this may take a while…"):
        start = time.time()
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.time() - start

    st.success(f"Finished in {elapsed:.1f} s (exit‑code {proc.returncode})")

    st.subheader("Execution Logs (scrollable)")
    log_tabs = st.tabs(["stdout", "stderr"])
    with log_tabs[0]:
        st.code(proc.stdout or "<empty>", language="bash")
    with log_tabs[1]:
        st.code(proc.stderr or "<empty>", language="bash")

    ######################################################################################
    # Two‑step video synthesis (PNG → FFV1 → H.264) – CPU‑only fast path
    ######################################################################################

    name = pathlib.Path(src_dir).stem
    frames_dir = os.path.join(output_dir, "primitives", name, f"synthesis/imitations/{name}-{pathlib.Path(ref_path_simple).name}")
    audio_path = os.path.join(output_dir, "primitives", pathlib.Path(ref_path_simple).name, "processed/audio.mp3")

    if os.path.isdir(frames_dir):
        fps = ref_fps or 30
        temp_avi = os.path.join(frames_dir, "temp.avi")
        out_mp4 = f"{frames_dir}_out.mp4"

        cmd1 = (
            f"ffmpeg -y -framerate {fps} "
            f"-pattern_type glob -i '{frames_dir}/pred_*.png' "
            f"-c:v ffv1 -level 3 -g 1 -threads 0 -pix_fmt yuv420p "
            f"{temp_avi}"
        )

        if os.path.exists(audio_path):
            cmd2 = (
                f"ffmpeg -y -i {temp_avi} -i {audio_path} "
                f"-c:v libx264 -preset ultrafast -crf 18 -threads 0 -pix_fmt yuv420p "
                f"-c:a copy -shortest -r {fps} {out_mp4}"
            )
        else:
            cmd2 = (
                f"ffmpeg -y -i {temp_avi} "
                f"-c:v libx264 -preset ultrafast -crf 18 -threads 0 -pix_fmt yuv420p "
                f"-r {fps} {out_mp4}"
            )

        st.subheader("Video synthesis commands")
        st.code(cmd1 + "\n" + cmd2, language="bash")

        with st.spinner("Creating video from frames (2‑step)…"):
            p1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
            if p1.returncode != 0:
                st.error(f"Step‑1 failed: {p1.stderr}")
                st.stop()
            p2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
            if p2.returncode != 0:
                st.error(f"Step‑2 failed: {p2.stderr}")
                st.stop()

        st.subheader("Result video")
        st.video(out_mp4)
    else:
        st.warning("frames_dir does not exist – nothing to encode.")
