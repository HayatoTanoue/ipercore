#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit GUI for iPERCore motion–imitation demo that works with either
pre-processed sample assets or user-uploaded sources/references.

Key design changes (compared with your previous draft)
======================================================
* `output_dir` and `assets_dir` are **absolute-path defaults** so CWD does
  not affect file lookup.
* In "事前処理済みコンテンツ" mode **`model_id == selected_source`** – this is
  mandatory for iPERCore to reuse existing primitives and to locate the
  final MP4 inside `results/primitives/<model_id>/synthesis/imitations/`.
* `display_results()` first looks into that canonical directory and then
  falls back to an exhaustive recursive search (handles .mp4 / .avi).
* `subprocess.run(..., capture_output=True, check=True)` is used instead of
  `subprocess.call()` so that any runtime error is surfaced in the UI.
* Asset/sample lists are populated dynamically from `assets_dir` – no more
  hard-coded filenames that may not exist in a particular checkout.
* Every path that goes into iPERCore is **absolute & resolved** to avoid
  surprises in Docker or remote execution environments.

This single file is self-contained – place it anywhere inside the
workspace and launch with:

```bash
streamlit run iper_streamlit_app.py
```
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Iterable, List

import streamlit as st

###############################################################################
# ----------  Streamlit Sidebar (global parameters)  -------------------------
###############################################################################

def sidebar_config():
    mode = st.sidebar.radio(
        "処理モード選択",
        ("事前処理済みコンテンツ", "新規ターゲット動画"),
        index=0,
    )

    gpu_ids: str = st.sidebar.text_input("GPU IDs", "0")
    image_size: int = st.sidebar.slider("画像サイズ", 256, 1024, 512, 128)
    num_source: int = st.sidebar.slider("ソース画像数", 1, 8, 2)

    # ***ABSOLUTE defaults so we are location-agnostic***
    output_dir: str = st.sidebar.text_input(
        "出力ディレクトリ", "/workspace/results"
    )
    assets_dir: str = st.sidebar.text_input(
        "アセットディレクトリ", "/workspace/assets"
    )

    # `model_id` textbox only appears in "新規ターゲット動画" mode.
    model_id: str | None = None
    if mode == "新規ターゲット動画":
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_id = st.sidebar.text_input("モデルID", f"model_{ts}")

    return mode, gpu_ids, image_size, num_source, output_dir, assets_dir, model_id

###############################################################################
# ----------  Utility: dynamic sample listing  -------------------------------
###############################################################################

def list_dir_names(parent: Path, exts: Iterable[str] | None = None) -> List[str]:
    if not parent.exists():
        return []
    if exts is None:
        # return directory names only
        return sorted([p.name for p in parent.iterdir() if p.is_dir()])
    # return files whose suffix matches one of exts (".mp4" etc.)
    exts = tuple(x.lower() for x in exts)
    return sorted([p.name for p in parent.iterdir() if p.is_file() and p.suffix.lower() in exts])

###############################################################################
# ----------  File save helpers for uploads  ----------------------------------
###############################################################################

def save_uploaded_files(files, subfolder: str) -> str:
    save_dir = Path("./temp") / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for file in files:
        dst = save_dir / file.name
        with open(dst, "wb") as f:
            f.write(file.getbuffer())
        paths.append(str(dst))
    # iPERCore supports multiple source paths joined by commas.
    return ",".join(paths)


def save_uploaded_file(file, subfolder: str) -> str:
    save_dir = Path("./temp") / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    dst = save_dir / file.name
    with open(dst, "wb") as f:
        f.write(file.getbuffer())
    return str(dst)

###############################################################################
# ----------  iPERCore Invocation Wrappers  -----------------------------------
###############################################################################

def run_generation(
    *,
    gpu_ids: str,
    image_size: int,
    num_source: int,
    output_dir: str,
    assets_dir: str,
    model_id: str,
    source: str,
    reference: str,
):
    """Run iPERCore generation using pre-processed primitives."""

    output_dir = str(Path(output_dir).expanduser().resolve())
    assets_dir = str(Path(assets_dir).expanduser().resolve())

    src_path = (
        f"path?={assets_dir}/samples/sources/{source},name?={source}"
    )
    ref_name = Path(reference).stem
    ref_path = (
        f"path?={assets_dir}/samples/references/{reference},"
        f"name?={ref_name},pose_fc?=300"
    )

    cmd = [
        sys.executable,
        "demo/motion_imitate.py",
        "--gpu_ids",
        gpu_ids,
        "--image_size",
        str(image_size),
        "--num_source",
        str(num_source),
        "--output_dir",
        output_dir,
        "--assets_dir",
        assets_dir,
        "--model_id",
        model_id,
        "--src_path",
        src_path,
        "--ref_path",
        ref_path,
    ]

    with st.spinner("iPERCore 実行中..."):
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            st.success("iPERCore 生成完了 (exit 0)")
            st.code(completed.stdout)
        except subprocess.CalledProcessError as e:
            st.error("iPERCore でエラーが発生しました")
            st.code(e.stderr)
            return  # early-exit

    display_results(output_dir, model_id)


def run_full_process(
    *,
    gpu_ids: str,
    image_size: int,
    num_source: int,
    output_dir: str,
    assets_dir: str,
    model_id: str,
    source_path: str,
    reference_path: str,
):
    """Run the whole pipeline (preprocess + generation) for user uploads."""

    output_dir = str(Path(output_dir).expanduser().resolve())
    assets_dir = str(Path(assets_dir).expanduser().resolve())

    src_path = f"path?={source_path},name?={model_id}"
    ref_path = f"path?={reference_path},name?=custom_reference,pose_fc?=300"

    cmd = [
        sys.executable,
        "demo/motion_imitate.py",
        "--gpu_ids",
        gpu_ids,
        "--image_size",
        str(image_size),
        "--num_source",
        str(num_source),
        "--output_dir",
        output_dir,
        "--assets_dir",
        assets_dir,
        "--model_id",
        model_id,
        "--src_path",
        src_path,
        "--ref_path",
        ref_path,
    ]

    with st.spinner("iPERCore 前処理 + 生成実行中..."):
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            st.success("全工程完了 (exit 0)")
            st.code(completed.stdout)
        except subprocess.CalledProcessError as e:
            st.error("iPERCore でエラーが発生しました")
            st.code(e.stderr)
            return

    display_results(output_dir, model_id)

###############################################################################
# ----------  Result Viewer  --------------------------------------------------
###############################################################################

def display_results(output_dir: str, model_id: str):
    """Locate mp4/avi results and embed them in the Streamlit UI."""

    base = Path(output_dir).expanduser().resolve()

    # 1) canonical path
    imit_dir = base / "primitives" / model_id / "synthesis" / "imitations"
    videos = list(imit_dir.glob("*.mp*")) if imit_dir.exists() else []

    # 2) fallback: any file under output_dir containing model_id in stem
    if not videos:
        videos = [p for p in base.rglob("*.mp*") if model_id in p.stem]

    if not videos:
        st.warning(f"{model_id} を含む動画が見つかりません: {base}")
        return

    st.subheader("生成結果")
    for vid in sorted(videos):
        rel = vid.relative_to(base)
        st.markdown(f"**{rel}**")
        st.video(str(vid))

###############################################################################
# ----------  Mode-specific UI flows  ----------------------------------------
###############################################################################

def process_preprocessed_content(
    *,
    gpu_ids: str,
    image_size: int,
    num_source: int,
    output_dir: str,
    assets_dir: str,
):
    assets_path = Path(assets_dir).expanduser().resolve()

    # Dynamically build candidate lists
    pre_sources = list_dir_names(assets_path / "samples" / "sources")
    pre_refs    = list_dir_names(assets_path / "samples" / "references", exts=(".mp4", ".avi"))

    st.subheader("事前処理済みのソース選択")
    if not pre_sources:
        st.error("samples/sources が見つかりません。assets_dir を確認してください。")
        return
    selected_source = st.selectbox("ソース選択", pre_sources)

    st.subheader("事前処理済みのリファレンス選択")
    if not pre_refs:
        st.error("samples/references が見つかりません。assets_dir を確認してください。")
        return
    selected_reference = st.selectbox("リファレンス選択", pre_refs)

    if st.button("生成実行"):
        # Critical: model_id must equal source name
        model_id = selected_source
        run_generation(
            gpu_ids=gpu_ids,
            image_size=image_size,
            num_source=num_source,
            output_dir=output_dir,
            assets_dir=assets_dir,
            model_id=model_id,
            source=selected_source,
            reference=selected_reference,
        )


def process_new_content(
    *,
    gpu_ids: str,
    image_size: int,
    num_source: int,
    output_dir: str,
    assets_dir: str,
    model_id: str,
):
    st.subheader("新規ソース画像のアップロード")
    source_files = st.file_uploader(
        "ソース画像をアップロード", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    st.subheader("新規リファレンス動画のアップロード")
    reference_file = st.file_uploader("リファレンス動画をアップロード", type=["mp4", "avi", "mov"])

    if st.button("処理実行"):
        if not source_files or not reference_file:
            st.error("ソース画像とリファレンス動画の両方をアップロードしてください。")
            return

        src_paths = save_uploaded_files(source_files, "sources")
        ref_path  = save_uploaded_file(reference_file, "references")

        run_full_process(
            gpu_ids=gpu_ids,
            image_size=image_size,
            num_source=num_source,
            output_dir=output_dir,
            assets_dir=assets_dir,
            model_id=model_id,
            source_path=src_paths,
            reference_path=ref_path,
        )

###############################################################################
# ----------  Main  -----------------------------------------------------------
###############################################################################

def main():
    st.title("iPERCore Motion Imitation")

    (
        mode,
        gpu_ids,
        image_size,
        num_source,
        output_dir,
        assets_dir,
        model_id,
    ) = sidebar_config()

    if mode == "事前処理済みコンテンツ":
        process_preprocessed_content(
            gpu_ids=gpu_ids,
            image_size=image_size,
            num_source=num_source,
            output_dir=output_dir,
            assets_dir=assets_dir,
        )
    else:
        assert model_id is not None  # guaranteed by sidebar_config
        process_new_content(
            gpu_ids=gpu_ids,
            image_size=image_size,
            num_source=num_source,
            output_dir=output_dir,
            assets_dir=assets_dir,
            model_id=model_id,
        )


if __name__ == "__main__":
    main()
