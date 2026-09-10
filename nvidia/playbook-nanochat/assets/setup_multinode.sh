#!/bin/bash

export HOST_IP=$1
export WORKER_IP=$2

workdir=$(pwd)

cmd="cd $workdir && \
git clone https://github.com/karpathy/nanochat.git && \
cd nanochat && \
git checkout c6b7ab744055d5915e6ccb61088de80c10cbaff9 && \
cp ../speedrun_multinode.sh ./speedrun.sh && \
cd .. && \
docker build -f Dockerfile.multinode -t nanochat ."

sh -c "$cmd"
ssh $USER@$WORKER_IP "$cmd"