#!/usr/bin/env bash
set -ex
echo "--- Building and pushing NPU Test Image"
echo "Image: ${IMAGE_REGISTRY}/${IMAGE_NAME}:${VIME_IMAGE_TAG}"
echo "buildkitd address: ${BUILDKITD_ADDR}"

if ! command -v buildctl > /dev/null 2>&1; then
  echo "Installing buildctl..."
  mkdir -p /tmp/buildkit
  BUILDKIT_VERSION="v0.29.0"
  curl -fsSL "https://gh-proxy.test.osinfra.cn/https://github.com/moby/buildkit/releases/download/${BUILDKIT_VERSION}/buildkit-${BUILDKIT_VERSION}.linux-arm64.tar.gz" -o /tmp/buildkit.tar.gz
  tar -xzf /tmp/buildkit.tar.gz -C /tmp/buildkit
  cp /tmp/buildkit/bin/buildctl /usr/local/bin/
fi

sed -i '/^RUN git config --global http.sslVerify false/i RUN git config --global url."https://gh-proxy.test.osinfra.cn/https://github.com/".insteadOf "https://github.com/"' docker/Dockerfile.npu
sed -i \
  's|git remote add origin https://github\.com/vllm-project/router\.git|git remote add origin https://gh-proxy.test.osinfra.cn/https://github.com/vllm-project/router.git|' \
  docker/Dockerfile.npu
sed -i '/^# syntax=docker\/dockerfile:1\.7$/d' docker/Dockerfile.npu

export DOCKER_CONFIG=/home/user/.docker
BASE_IMAGE_TAG="$(sed -n 's/^ARG BASE_IMAGE_TAG=//p' docker/Dockerfile.npu)"
test -n "${BASE_IMAGE_TAG}"
# The final component of the Dockerfile tag is the hardware variant.
[[ "$BASE_IMAGE_TAG" == *-* ]]
BASE_IMAGE_TAG="${BASE_IMAGE_TAG%-*}-$NPU_IMAGE_VARIANT"
soc_args=()
if [[ -n "${NPU_SOC_VERSION:-}" ]]; then
  soc_args+=(--opt "build-arg:SOC_VERSION=$NPU_SOC_VERSION")
fi
BASE_IMAGE_PROBE_DIR="$(mktemp -d)"
trap 'rm -rf "${BASE_IMAGE_PROBE_DIR}"' EXIT
printf 'FROM %s:%s\n' "${BASE_IMAGE}" "${BASE_IMAGE_TAG}" > "${BASE_IMAGE_PROBE_DIR}/Dockerfile"

# Fall back to the Dockerfile default when the internal tag is unavailable.
if ! timeout 60 buildctl \
  --addr="${BUILDKITD_ADDR}" \
  --tlscacert=/home/user/.docker/ca.pem \
  --tlscert=/home/user/.docker/cert.pem \
  --tlskey=/home/user/.docker/key.pem \
  build \
  --frontend dockerfile.v0 \
  --local context="${BASE_IMAGE_PROBE_DIR}" \
  --local dockerfile="${BASE_IMAGE_PROBE_DIR}" \
  --opt image-resolve-mode=pull \
  --progress=plain; then
  BASE_IMAGE="$(sed -n 's/^ARG BASE_IMAGE=//p' docker/Dockerfile.npu)"
  if [[ -z "${BASE_IMAGE}" ]]; then
    echo "ERROR: Missing or empty ARG BASE_IMAGE in docker/Dockerfile.npu; cannot select a fallback image" >&2
    exit 1
  fi
  echo "Internal base image unavailable; using ${BASE_IMAGE}:${BASE_IMAGE_TAG}"
fi

buildctl \
  --addr="${BUILDKITD_ADDR}" \
  --tlscacert=/home/user/.docker/ca.pem \
  --tlscert=/home/user/.docker/cert.pem \
  --tlskey=/home/user/.docker/key.pem \
  build \
  --frontend dockerfile.v0 \
  --local context=. \
  --local dockerfile=./docker \
  --opt filename=Dockerfile.npu \
  --opt build-arg:BASE_IMAGE="${BASE_IMAGE}" \
  --opt build-arg:BASE_IMAGE_TAG="${BASE_IMAGE_TAG}" \
  "${soc_args[@]}" \
  --opt build-arg:VIME_COMMIT="${BUILDKITE_COMMIT}" \
  --opt build-arg:RUST_IMAGE="${RUST_IMAGE}" \
  --opt build-arg:CARGO_REGISTRY="${CARGO_REGISTRY}" \
  --secret id=dockerconfig,src=/home/user/.docker/config.json \
  --output type=image,name=${IMAGE_REGISTRY}/${IMAGE_NAME}:${VIME_IMAGE_TAG},push=true \
  --progress=plain

echo "--- Image pushed successfully"
echo "${IMAGE_REGISTRY}/${IMAGE_NAME}:${VIME_IMAGE_TAG}"
