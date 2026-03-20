#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-parameter-golf-smoke}"
GPU_TYPE="${GPU_TYPE:-NVIDIA A40}"
GPU_COUNT="${GPU_COUNT:-1}"
IMAGE_NAME="${IMAGE_NAME:-runpod/parameter-golf:latest}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-60}"
VOLUME_GB="${VOLUME_GB:-60}"
MIN_VCPU="${MIN_VCPU:-8}"
MIN_MEM_GB="${MIN_MEM_GB:-32}"
COST_LIMIT="${COST_LIMIT:-0.30}"

runpodctl create pod \
  --communityCloud \
  --name "${NAME}" \
  --imageName "${IMAGE_NAME}" \
  --gpuType "${GPU_TYPE}" \
  --gpuCount "${GPU_COUNT}" \
  --containerDiskSize "${CONTAINER_DISK_GB}" \
  --volumeSize "${VOLUME_GB}" \
  --vcpu "${MIN_VCPU}" \
  --mem "${MIN_MEM_GB}" \
  --cost "${COST_LIMIT}" \
  --ports 8888/http \
  --ports 22/tcp
