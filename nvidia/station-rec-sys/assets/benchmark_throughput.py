"""Throughput benchmark for the recommendation endpoint.

Fires concurrent requests at /api/recommend to measure latency and throughput.
Uses asyncio + aiohttp for truly concurrent requests.

Usage:
  # Start the app first: python app.py
  # Then: python scripts/benchmark_throughput.py --concurrency 100 --requests 1000
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp


async def fetch_recommendation(session, url, user_id, results):
    """Single recommendation request."""
    t0 = time.perf_counter()
    try:
        async with session.get(f"{url}/api/recommend/{user_id}") as resp:
            data = await resp.json()
            latency = (time.perf_counter() - t0) * 1000
            results.append({
                'latency_ms': latency,
                'status': resp.status,
                'method': data.get('method', 'unknown'),
                'server_latency_ms': data.get('latency_ms', 0),
            })
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        results.append({
            'latency_ms': latency,
            'status': 0,
            'error': str(e),
        })


async def run_benchmark(url, user_ids, concurrency, total_requests):
    """Run the benchmark with the given concurrency level."""
    results = []
    sem = asyncio.Semaphore(concurrency)

    async def bounded_fetch(session, uid):
        async with sem:
            await fetch_recommendation(session, url, uid, results)

    connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(total_requests):
            uid = user_ids[i % len(user_ids)]
            tasks.append(bounded_fetch(session, uid))

        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    return results, wall_time


def print_results(results, wall_time, concurrency, total_requests):
    """Print benchmark results with latency percentiles."""
    successful = [r for r in results if r.get('status') == 200]
    failed = [r for r in results if r.get('status') != 200]
    latencies = [r['latency_ms'] for r in successful]
    server_latencies = [r['server_latency_ms'] for r in successful if r.get('server_latency_ms')]

    print(f"\n{'='*60}")
    print(f"THROUGHPUT BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Concurrency:     {concurrency}")
    print(f"Total requests:  {total_requests}")
    print(f"Successful:      {len(successful)}")
    print(f"Failed:          {len(failed)}")
    print(f"Wall time:       {wall_time:.2f}s")
    print(f"Throughput:      {len(successful)/wall_time:.0f} req/sec")

    if latencies:
        latencies.sort()
        print(f"\nClient-side latency (includes network):")
        print(f"  Min:    {min(latencies):.1f}ms")
        print(f"  p50:    {latencies[len(latencies)//2]:.1f}ms")
        print(f"  p95:    {latencies[int(len(latencies)*0.95)]:.1f}ms")
        print(f"  p99:    {latencies[int(len(latencies)*0.99)]:.1f}ms")
        print(f"  Max:    {max(latencies):.1f}ms")
        print(f"  Mean:   {statistics.mean(latencies):.1f}ms")

    if server_latencies:
        server_latencies.sort()
        print(f"\nServer-side latency (recommendation only):")
        print(f"  Min:    {min(server_latencies):.0f}ms")
        print(f"  p50:    {server_latencies[len(server_latencies)//2]:.0f}ms")
        print(f"  p95:    {server_latencies[int(len(server_latencies)*0.95)]:.0f}ms")
        print(f"  p99:    {server_latencies[int(len(server_latencies)*0.99)]:.0f}ms")
        print(f"  Max:    {max(server_latencies):.0f}ms")

    methods = {}
    for r in successful:
        m = r.get('method', 'unknown')
        methods[m] = methods.get(m, 0) + 1
    if methods:
        print(f"\nMethods: {methods}")

    print(f"{'='*60}")

    return {
        'concurrency': concurrency,
        'total_requests': total_requests,
        'successful': len(successful),
        'failed': len(failed),
        'wall_time_s': wall_time,
        'throughput_rps': len(successful) / wall_time,
        'p50_ms': latencies[len(latencies)//2] if latencies else 0,
        'p95_ms': latencies[int(len(latencies)*0.95)] if latencies else 0,
        'p99_ms': latencies[int(len(latencies)*0.99)] if latencies else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description='Benchmark recommendation throughput')
    parser.add_argument('--url', default='http://localhost:7860', help='App URL')
    parser.add_argument('--concurrency', type=int, default=100, help='Concurrent requests')
    parser.add_argument('--requests', type=int, default=1000, help='Total requests')
    parser.add_argument('--sweep', action='store_true', help='Sweep concurrency levels')
    args = parser.parse_args()

    # Get user IDs from the app
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{args.url}/api/users") as resp:
            users = await resp.json()
    user_ids = [u['id'] for u in users]
    print(f"Loaded {len(user_ids)} user IDs")

    if args.sweep:
        # Sweep concurrency levels
        all_results = []
        for conc in [1, 10, 50, 100, 500, 1000]:
            n = max(conc * 5, 500)
            print(f"\n--- Concurrency: {conc}, Requests: {n} ---")
            results, wall_time = await run_benchmark(args.url, user_ids, conc, n)
            stats = print_results(results, wall_time, conc, n)
            all_results.append(stats)

        # Summary table
        print(f"\n{'='*60}")
        print(f"SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"{'Concurrency':>12} {'Throughput':>12} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10}")
        print(f"{'-'*60}")
        for s in all_results:
            print(f"{s['concurrency']:>12} {s['throughput_rps']:>10.0f}/s {s['p50_ms']:>10.1f} {s['p95_ms']:>10.1f} {s['p99_ms']:>10.1f}")
    else:
        results, wall_time = await run_benchmark(args.url, user_ids, args.concurrency, args.requests)
        print_results(results, wall_time, args.concurrency, args.requests)


if __name__ == '__main__':
    asyncio.run(main())
