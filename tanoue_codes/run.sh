# 割り当てGPUを選択できるようにする(複数)
USE_GPU_IDS=${1:-0}

# docker run コマンドを実行するスクリプト
docker run --gpus '"device='"$USE_GPU_IDS"'"' \
    --rm \
    -it \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v ${NAS_PATH}:${NAS_PATH} \
    -v ${PWD}:/workspace \
    --net=host \
    --shm-size=200G \
    ipercore:latest bash -c "cd /workspace && bash"