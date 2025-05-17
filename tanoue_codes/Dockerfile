###############################################################################
# Dockerfile – iPERCore (CUDA-selectable, Torch-selectable)
#   * Ubuntu 18.04 / CUDA 10.2.89  or 11.0.3 or 11.1.1
#   * Python 3.8 + venv, gcc-7.5, ffmpeg-4.4, git
#   * Torch / torchvision / mmcv-full を CUDA に合わせ自動インストール
#   * iPERCore 本体を python setup.py develop で組み込み
###############################################################################
#========== 1. ビルド時パラメータ (必要に応じて --build-arg で上書き) =========#
ARG CUDA_VERSION=11.1.1          # 10.2.89 / 11.0.3 / 11.1.1
ARG TORCH_VERSION=1.8.1          # Ampere→1.8.1, それ以外→1.7.0
ARG TORCH_CUDA_TAG=cu111         # cu102 / cu110 / cu111
ARG TORCHVISION_VERSION=0.9.1    # torch1.8.1 → 0.9.1, torch1.7.0 → 0.8.1
ARG MMCV_VERSION=1.3.18          # torch1.8.1 用, torch1.7.0 なら 1.2.0
ARG UBUNTU_BASE=18.04

#========== 2. ベース CUDA イメージ (パッチ番号必須) ==========================#
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_BASE}

#========== 3. 共通環境変数 ===================================================#
ARG TORCH_VERSION
ARG TORCH_CUDA_TAG
ARG TORCHVISION_VERSION
ARG MMCV_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6+PTX 7.5+PTX 7.0+PTX" \
    TORCH_VERSION=${TORCH_VERSION} \
    TORCH_CUDA_TAG=${TORCH_CUDA_TAG} \
    TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    MMCV_VERSION=${MMCV_VERSION}

#========== 4. OS パッケージ & Python 3.8 ====================================#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common curl gnupg2 ca-certificates \
    build-essential gcc-7 g++-7 git ninja-build \
    ffmpeg && \
    add-apt-repository -y ppa:savoury1/ffmpeg4 && \
    apt-get update && apt-get -y upgrade ffmpeg && \
    apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-venv python3-distutils python3-pip && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#========== 5. Python 仮想環境 (venv) ========================================#
RUN pip install --no-cache-dir virtualenv && \
    python -m virtualenv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

#========== 6. 深層学習ライブラリ ============================================#
# (A) torch / torchvision -----------------------------------------------------#
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${TORCH_CUDA_TAG} \
    torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG} \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir \
    cython==0.29.36

# (B) mmcv-full --------------------------------------------------------------#
RUN pip install --no-cache-dir \
    mmcv-full==${MMCV_VERSION} \
    -f https://download.openmmlab.com/mmcv/dist/${TORCH_CUDA_TAG}/torch${TORCH_VERSION%.*}.0/index.html

# (C) install.md の残り依存 + ビルドツール -----------------------------------#
RUN pip install --no-cache-dir \
    numpy>=1.19.3 \
    scipy>=1.5.2 \
    scikit-image>=0.17.2 \
    opencv-python>=4.4.0.46 \
    tensorboardX>=2.1 \
    tqdm>=4.48.2 \
    visdom>=0.1.8.9 \
    easydict>=1.9 \
    toml>=0.10.2 \
    git+https://github.com/open-mmlab/mmdetection.git@8179440ec5f75fe95484854af61ce6f6279f3bbc \
    git+https://github.com/open-mmlab/mmediting.git@d4086aaf8a36ae830f1714aad585900d24ad1156 \
    git+https://github.com/iPERDance/neural_renderer.git@e5f54f71a8941acf372514eb92e289872f272653

#========== 7. iPERCore ソースを組み込み =====================================#
RUN mkdir -p /ipercpre_build
WORKDIR /ipercpre_build
RUN git clone https://github.com/iPERDance/iPERCore.git
WORKDIR /ipercpre_build/iPERCore
RUN python setup.py develop

# clean up existing files

#========== 8. デフォルト CMD ================================================#
CMD ["/bin/bash"]
