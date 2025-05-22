#!/usr/bin/env python3
"""
Enhanced Streamlit GUI for iPERCore motion–imitation demo with improved video display.

Key improvements:
* Flexible output directory handling for both Docker and non-Docker environments
* Better video discovery with detailed path information
* Enhanced video display with metadata
* Support for both absolute and relative paths
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
import glob
from pathlib import Path
from typing import Iterable, List, Optional

import streamlit as st


def sidebar_config():
    mode = st.sidebar.radio(
        "処理モード選択",
        ("事前処理済みコンテンツ", "新規ターゲット動画"),
        index=0,
    )

    gpu_ids: str = st.sidebar.text_input("GPU IDs", "0")
    image_size: int = st.sidebar.slider("画像サイズ", 256, 1024, 512, 128)
    num_source: int = st.sidebar.slider("ソース画像数", 1, 8, 2)

    st.sidebar.markdown("### 出力設定")
    st.sidebar.markdown("""
    **注意**: Docker内では `/workspace/results` を使用し、
    Docker外では相対パス `./results` を使用することをお勧めします。
    """)
    
    output_dir: str = st.sidebar.text_input(
        "出力ディレクトリ", "/workspace/results"
    )
    
    if not os.path.exists("/workspace"):
        st.sidebar.warning("Dockerコンテナ外で実行している可能性があります。相対パス './results' の使用を検討してください。")
    
    assets_dir: str = st.sidebar.text_input(
        "アセットディレクトリ", "/workspace/assets"
    )
    
    if not os.path.exists("/workspace"):
        st.sidebar.warning("Dockerコンテナ外で実行している可能性があります。相対パス './assets' の使用を検討してください。")

    model_id: str | None = None
    if mode == "新規ターゲット動画":
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_id = st.sidebar.text_input("モデルID", f"model_{ts}")

    return mode, gpu_ids, image_size, num_source, output_dir, assets_dir, model_id


def list_dir_names(parent: Path, exts: Iterable[str] | None = None) -> List[str]:
    if not parent.exists():
        return []
    if exts is None:
        return sorted([p.name for p in parent.iterdir() if p.is_dir()])
    exts = tuple(x.lower() for x in exts)
    return sorted([p.name for p in parent.iterdir() if p.is_file() and p.suffix.lower() in exts])


def save_uploaded_files(files, subfolder: str) -> str:
    save_dir = Path("./temp") / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for file in files:
        dst = save_dir / file.name
        with open(dst, "wb") as f:
            f.write(file.getbuffer())
        paths.append(str(dst))
    return ",".join(paths)


def save_uploaded_file(file, subfolder: str) -> str:
    save_dir = Path("./temp") / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    dst = save_dir / file.name
    with open(dst, "wb") as f:
        f.write(file.getbuffer())
    return str(dst)


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


def find_videos(base_dir: Path, model_id: str) -> List[Path]:
    """
    Find videos in multiple possible locations with detailed logging.
    """
    videos = []
    
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        st.info(f"出力ディレクトリを作成しました: {base_dir}")
        return videos
    
    imit_dir = base_dir / "primitives" / model_id / "synthesis" / "imitations"
    if imit_dir.exists():
        mp4_videos = list(imit_dir.glob("*.mp4"))
        avi_videos = list(imit_dir.glob("*.avi"))
        videos.extend(mp4_videos)
        videos.extend(avi_videos)
        if mp4_videos or avi_videos:
            st.info(f"正規パスで動画が見つかりました: {imit_dir}")
    
    alt_paths = [
        base_dir / "primitives" / model_id / "synthesis",  # Check synthesis directory
        base_dir / "primitives" / model_id,                # Check model directory
        base_dir / "primitives",                           # Check primitives directory
        base_dir,                                          # Check base directory
    ]
    
    for path in alt_paths:
        if path.exists() and not videos:  # Only search if we haven't found videos yet
            for video in path.glob("**/*.mp4"):
                if model_id in video.stem:
                    videos.append(video)
            for video in path.glob("**/*.avi"):
                if model_id in video.stem:
                    videos.append(video)
            
            if videos:
                st.info(f"代替パスで動画が見つかりました: {path}")
                break
    
    return videos

def display_results(output_dir: str, model_id: str):
    """Locate mp4/avi results and embed them in the Streamlit UI with enhanced information."""

    base = Path(output_dir).expanduser().resolve()
    
    st.info(f"出力ディレクトリ: {base}")
    
    videos = find_videos(base, model_id)

    if not videos:
        st.warning(f"{model_id} を含む動画が見つかりません: {base}")
        
        st.markdown("""
        
        1. 出力ディレクトリが正しく設定されているか確認してください
        2. Docker内で実行している場合は `/workspace/results` を使用してください
        3. Docker外で実行している場合は相対パス `./results` を使用してください
        4. 動画生成が正常に完了したか確認してください
        
        ```
        {output_dir}/primitives/{model_id}/synthesis/imitations/{model_id}-{ref_name}.mp4
        ```
        """.format(output_dir=output_dir, model_id=model_id))
        return

    st.subheader("生成結果")
    
    for i, vid in enumerate(sorted(videos)):
        rel = vid.relative_to(base) if base in vid.parents else vid
        st.markdown(f"### 動画 {i+1}: {rel}")
        
        st.markdown(f"""
        **ファイル情報:**
        - 完全パス: `{vid}`
        - ファイル名: `{vid.name}`
        - サイズ: {vid.stat().st_size / (1024*1024):.2f} MB
        - 最終更新: {time.ctime(vid.stat().st_mtime)}
        """)
        
        st.video(str(vid))


def process_preprocessed_content(
    *,
    gpu_ids: str,
    image_size: int,
    num_source: int,
    output_dir: str,
    assets_dir: str,
):
    assets_path = Path(assets_dir).expanduser().resolve()

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


def main():
    st.title("iPERCore Motion Imitation")
    
    st.markdown("""
    このアプリケーションは、iPERCoreを使用して人物の動きを別の人物に転写するデモです。
    事前処理済みのコンテンツを使用するか、新しい画像と動画をアップロードして処理することができます。
    
    1. ソース画像（動きを転写される人物）を選択または提供
    2. リファレンス動画（動きの元となる人物）を選択または提供
    3. 「生成実行」または「処理実行」ボタンをクリック
    4. 処理が完了すると、生成された動画が下部に表示されます
    """)

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
    
    st.markdown("---")
    st.subheader("動画を手動で検索")
    
    with st.expander("動画検索オプション"):
        manual_output_dir = st.text_input("出力ディレクトリパス", output_dir)
        manual_model_id = st.text_input("モデルID", model_id or "")
        
        if st.button("検索"):
            if manual_model_id:
                display_results(manual_output_dir, manual_model_id)
            else:
                st.error("モデルIDを入力してください")


if __name__ == "__main__":
    main()
