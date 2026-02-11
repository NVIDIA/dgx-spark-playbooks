#!/usr/bin/env python3
"""
NCCL Communication Test Script

Tests NCCL (NVIDIA Collective Communications Library) communication over RDMA
between two nodes in a distributed setup.

Usage:
    On Node 0 (head): python test_nccl.py --rank 0
    On Node 1 (worker): python test_nccl.py --rank 1

Requirements:
    - PyTorch with CUDA support
    - NCCL backend available
    - RDMA network configured between nodes
"""

import os
import torch
import torch.distributed as dist
import argparse


def test_nccl_communication():
    parser = argparse.ArgumentParser(description='Test NCCL communication over RDMA')
    parser.add_argument('--rank', type=int, required=True,
                        help='Rank of this process (0 for head, 1 for worker)')
    parser.add_argument('--world_size', type=int, default=2,
                        help='Total number of processes')
    parser.add_argument('--master_addr', type=str, default='192.168.200.1',
                        help='IP address of the head node')
    parser.add_argument('--master_port', type=str, default='29500',
                        help='Port for distributed communication')
    parser.add_argument('--interface', type=str, default='enp1s0f0np0',
                        help='Network interface for NCCL socket')
    args = parser.parse_args()

    # Set environment variables for distributed communication
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['NCCL_SOCKET_IFNAME'] = args.interface

    print(f"=" * 60)
    print(f"NCCL Communication Test")
    print(f"=" * 60)
    print(f"Rank: {args.rank}")
    print(f"World Size: {args.world_size}")
    print(f"Master: {args.master_addr}:{args.master_port}")
    print(f"Interface: {args.interface}")
    print(f"=" * 60)

    print(f"\n[Rank {args.rank}] Initializing process group...")

    # Initialize the process group with NCCL backend
    dist.init_process_group(
        backend='nccl',
        rank=args.rank,
        world_size=args.world_size
    )

    print(f"[Rank {args.rank}] Process group initialized successfully!")
    print(f"[Rank {args.rank}] Distributed rank: {dist.get_rank()}/{dist.get_world_size()}")

    # Create a tensor on GPU
    device = torch.device('cuda:0')
    tensor = torch.ones(10, device=device) * (args.rank + 1)

    print(f"\n[Rank {args.rank}] Before all_reduce: {tensor.tolist()}")

    # Perform all-reduce operation (sum across all ranks)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[Rank {args.rank}] After all_reduce: {tensor.tolist()}")

    # Calculate expected result
    expected = sum(range(1, args.world_size + 1))
    expected_tensor = torch.ones(10) * expected
    print(f"[Rank {args.rank}] Expected result: {expected_tensor.tolist()}")

    # Verify result
    if torch.allclose(tensor.cpu(), expected_tensor):
        print(f"\n[Rank {args.rank}] ✓ All-reduce test PASSED!")
    else:
        print(f"\n[Rank {args.rank}] ✗ All-reduce test FAILED!")

    # Cleanup
    dist.destroy_process_group()

    print(f"[Rank {args.rank}] Test completed successfully!")
    print(f"=" * 60)


if __name__ == "__main__":
    test_nccl_communication()
