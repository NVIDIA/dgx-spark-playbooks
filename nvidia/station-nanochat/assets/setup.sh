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
git checkout c6b7ab744055d5915e6ccb61088de80c10cbaff9 && \
cp ../speedrun_spark.sh ./speedrun.sh && \
cd .. && \
chmod +x launch_full.sh 2>/dev/null || true && \
docker build -t nanochat ."

sh -c "$cmd"
