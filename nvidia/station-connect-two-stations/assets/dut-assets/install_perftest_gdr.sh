#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install or build perftest with CUDA DMA-BUF/Data Direct support.
#
# This script runs on one DUT. It is intentionally optional: use it only when
# Step 6 reports that the installed ib_write_bw lacks --use_cuda_dmabuf or
# --use_data_direct.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

CHECK_ONLY=0
SOURCE_ONLY=0
PERFTEST_REPO="${PERFTEST_REPO:-https://github.com/linux-rdma/perftest.git}"
PERFTEST_REF="${PERFTEST_REF:-26.04.17}"
PERFTEST_PREFIX="${PERFTEST_PREFIX:-/usr/local}"
PERFTEST_SRC_DIR="${PERFTEST_SRC_DIR:-/usr/local/src/cx8-perftest-gdr}"
export PATH="${PERFTEST_PREFIX}/bin:${PATH}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      ;;
    --source-only)
      SOURCE_ONLY=1
      ;;
    --ref)
      [[ -n "${2:-}" ]] || die "--ref requires a git ref"
      PERFTEST_REF="$2"
      shift
      ;;
    -h|--help)
      echo "Usage: sudo $0 [--check-only] [--source-only] [--ref GIT_REF]"
      exit 0
      ;;
    *)
      die "Usage: sudo $0 [--check-only] [--source-only] [--ref GIT_REF]"
      ;;
  esac
  shift
done

[[ "${EUID}" == "0" ]] || die "Run through sudo; this script installs packages under the DUT OS"

warn_count=0
warn() {
  warn_count=$((warn_count + 1))
  echo "WARN: $*"
}

info() {
  echo "INFO: $*"
}

perftest_help() {
  command -v ib_write_bw >/dev/null 2>&1 || return 1
  ib_write_bw --help 2>&1 || true
}

perftest_has_gdr_flags() {
  local help_text
  help_text="$(perftest_help)" || return 1
  grep -q -- '--use_cuda_dmabuf' <<<"${help_text}" && \
    grep -q -- '--use_data_direct' <<<"${help_text}"
}

print_perftest_identity() {
  if command -v ib_write_bw >/dev/null 2>&1; then
    info "ib_write_bw path: $(command -v ib_write_bw)"
    ib_write_bw --version 2>&1 | sed 's/^/INFO: ib_write_bw version: /' || true
  else
    info "ib_write_bw is not currently installed"
  fi
}

find_cuda_h() {
  if [[ -n "${CUDA_H_PATH:-}" && -f "${CUDA_H_PATH}" ]]; then
    echo "${CUDA_H_PATH}"
    return 0
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/include/cuda.h \
    /usr/include/cuda.h \
    /usr/local/cuda-*/include/cuda.h; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

apt_install() {
  command -v apt-get >/dev/null 2>&1 || return 1
  export DEBIAN_FRONTEND=noninteractive
  apt-get -q update
  apt-get -q install -y "$@"
}

try_os_package_upgrade() {
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get is not available; skipping OS package upgrade path"
    return 1
  fi
  note "Installing/upgrading OS perftest package"
  apt_install perftest rdma-core ibverbs-providers || return 1
  hash -r
  perftest_has_gdr_flags
}

build_from_source() {
  local cuda_h
  local cuda_lib_dir
  cuda_h="$(find_cuda_h)" || die "cuda.h not found. Install CUDA development headers or set CUDA_H_PATH=/path/to/cuda.h"

  if ! command -v apt-get >/dev/null 2>&1; then
    die "apt-get is not available and source build dependencies cannot be installed automatically"
  fi

  for cuda_lib_dir in /usr/local/cuda/lib64 /usr/local/cuda-*/lib64; do
    if [[ -d "${cuda_lib_dir}" ]]; then
      export LIBRARY_PATH="${cuda_lib_dir}:${LIBRARY_PATH:-}"
      export LD_LIBRARY_PATH="${cuda_lib_dir}:${LD_LIBRARY_PATH:-}"
      break
    fi
  done

  note "Installing source-build dependencies"
  apt_install \
    git ca-certificates build-essential autoconf automake libtool pkg-config \
    libibverbs-dev librdmacm-dev libibumad-dev libnuma-dev libpci-dev \
    libnl-3-dev libnl-route-3-dev pciutils

  note "Preparing perftest source at ${PERFTEST_SRC_DIR}"
  mkdir -p "$(dirname "${PERFTEST_SRC_DIR}")"
  if [[ -d "${PERFTEST_SRC_DIR}/.git" ]]; then
    git -C "${PERFTEST_SRC_DIR}" fetch --all --tags
  elif [[ -e "${PERFTEST_SRC_DIR}" ]]; then
    die "${PERFTEST_SRC_DIR} already exists but is not a git checkout; remove it manually or set PERFTEST_SRC_DIR to an empty path"
  else
    git clone "${PERFTEST_REPO}" "${PERFTEST_SRC_DIR}"
  fi
  git -C "${PERFTEST_SRC_DIR}" checkout "${PERFTEST_REF}"

  note "Building perftest ref ${PERFTEST_REF} with CUDA_H_PATH=${cuda_h}"
  (
    cd "${PERFTEST_SRC_DIR}"
    ./autogen.sh
    ./configure --prefix="${PERFTEST_PREFIX}" CUDA_H_PATH="${cuda_h}"
    make -j"$(nproc)"
    make install
  )
  hash -r
}

print_perftest_identity
if perftest_has_gdr_flags; then
  echo "PASS: installed ib_write_bw already supports --use_cuda_dmabuf and --use_data_direct"
  exit 0
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  die "installed ib_write_bw does not support --use_cuda_dmabuf and --use_data_direct"
fi

if [[ "${SOURCE_ONLY}" == "0" ]]; then
  if try_os_package_upgrade; then
    print_perftest_identity
    echo "PASS: OS perftest package now supports --use_cuda_dmabuf and --use_data_direct"
    exit 0
  fi
  warn "OS package path did not provide the required perftest flags; trying source build"
else
  info "Skipping OS package upgrade path because --source-only was requested"
fi

build_from_source
print_perftest_identity

if perftest_has_gdr_flags; then
  if [[ "${warn_count}" == "0" ]]; then
    echo "PASS: source-built ib_write_bw supports --use_cuda_dmabuf and --use_data_direct"
  else
    echo "PASS: source-built ib_write_bw supports --use_cuda_dmabuf and --use_data_direct with ${warn_count} warning(s)"
  fi
else
  die "installed ib_write_bw still lacks --use_cuda_dmabuf/--use_data_direct after source build; check rdma-core/libibverbs Data Direct support and the build log"
fi
