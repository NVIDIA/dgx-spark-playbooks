#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

workdir=$(pwd)
# Directory where this script lives (assets)
assets_dir="$(cd "$(dirname "$0")" && pwd)"

cmd="cd $workdir && \
git clone https://github.com/karpathy/nanochat.git && \
cd nanochat && \
git checkout 0aaca56805eb13f6e6e1fff789a08086902f12ab && \
cp ../speedrun_station.sh ./runs/speedrun.sh && \
cd .. && \
chmod +x launch.sh 2>/dev/null || true && \
docker build -t nanochat ."

sh -c "$cmd"
