#!/bin/bash

CONTAINER_NAME="zk-dla-0806"
IMAGE_NAME="harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10"
VOLUME_MOUNTS=(
    "-v /dev/infiniband:/dev/infiniband"
    "-v /opt/hyhal:/opt/hyhal:ro"
    "-v /data1:/data1"
    "-v /data:/data"
)

# 检查容器状态函数
check_container_state() {
    docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}' 2>/dev/null
}

# 主逻辑
if container_state=$(check_container_state); then
    echo "CONTAINER state:====== $container_state ======"

    if [[ "$container_state" == "exited" ]]; then
        if ! docker start "$CONTAINER_NAME"; then
            echo "Failed to start container $CONTAINER_NAME" >&2
            exit 1
        fi
    fi

    exec docker exec -it "$CONTAINER_NAME" /bin/bash
else
    if ! docker run -dit \
        --network=host \
        --name="$CONTAINER_NAME" \
        --privileged \
        --device=/dev/kfd \
        --device=/dev/dri \
        --device=/dev/mkfd \
        --ipc=host \
        --shm-size=256G \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -u root \
        --ulimit stack=-1:-1 \
        --ulimit memlock=-1:-1 \
        -v /dev/infiniband:/dev/infiniband \
        -v /opt/hyhal:/opt/hyhal:ro \
        -v /data1:/data1 \
        -v /data:/data \
        "$IMAGE_NAME"
    then
        echo "Failed to create container $CONTAINER_NAME" >&2
        exit 1
    fi

    exec docker exec -it "$CONTAINER_NAME" /bin/bash
fi
