import argparse
import csv
import os
from pathlib import Path
from mpi4py import MPI


def get_iters(size, base_iters):
    # scale down iterations for huge messages so it doesn't take forever
    if size <= 1024:
        return base_iters
    if size <= 65536:
        return max(base_iters // 4, 1)
    return max(base_iters // 20, 1)


def run_bench(comm, msg_size, iters):
    rank = comm.Get_rank()
    payload = b"x" * msg_size

    comm.Barrier()
    start_t = MPI.Wtime()

    if rank == 0:
        for _ in range(iters):
            comm.ssend(payload, dest=1, tag=10)
            comm.recv(source=1, tag=11)
    elif rank == 1:
        for _ in range(iters):
            data = comm.recv(source=0, tag=10)
            comm.ssend(data, dest=0, tag=11)

    comm.Barrier()
    if rank == 0:
        return MPI.Wtime() - start_t
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="0,1,8,64,512,4096,32768,262144,1048576")
    parser.add_argument("--base-iterations", type=int, default=10000)
    parser.add_argument("--output", default="results/ping_pong.csv")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    results = []

    for size in sizes:
        iters = get_iters(size, args.base_iterations)
        elapsed = run_bench(comm, size, iters)

        if rank == 0 and elapsed is not None:
            total_msgs = 2 * iters
            t_per_msg = elapsed / total_msgs
            
            bw = 0.0
            if size > 0:
                bw = (size / 1_000_000.0) / t_per_msg

            results.append({
                "size_bytes": str(size),
                "iterations": str(iters),
                "total_time_seconds": f"{elapsed:.6f}",
                "time_per_message_seconds": f"{t_per_msg:.9f}",
                "bandwidth_mb_s": f"{bw:.6f}",
            })
            print(f"{size:>8} bytes, {iters:>6} iterations, {t_per_msg:.9f} s/message")

    if rank == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "size_bytes",
                    "iterations",
                    "total_time_seconds",
                    "time_per_message_seconds",
                    "bandwidth_mb_s",
                ],
            )
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
