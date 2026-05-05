import argparse
import csv
from pathlib import Path

from mpi4py import MPI


DEFAULT_SIZES = "0,1,8,64,512,4096,32768,262144,1048576"


def parse_sizes(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def iterations_for_size(size, base_iterations):
    if size <= 1024:
        return base_iterations
    if size <= 65536:
        return max(base_iterations // 4, 1)
    return max(base_iterations // 20, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--base-iterations", type=int, default=10000)
    parser.add_argument("--output", default="results/ping_pong.csv")
    return parser.parse_args()


def run_benchmark(comm, message_size, iterations):
    rank = comm.Get_rank()
    payload = b"x" * message_size

    comm.Barrier()
    start = MPI.Wtime()

    if rank == 0:
        for _ in range(iterations):
            comm.ssend(payload, dest=1, tag=10)
            comm.recv(source=1, tag=11)
    elif rank == 1:
        for _ in range(iterations):
            received = comm.recv(source=0, tag=10)
            comm.ssend(received, dest=0, tag=11)

    comm.Barrier()
    if rank == 0:
        return MPI.Wtime() - start
    return None


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sizes = parse_sizes(args.sizes)
    rows = []

    for message_size in sizes:
        iterations = iterations_for_size(message_size, args.base_iterations)
        elapsed = run_benchmark(comm, message_size, iterations)

        if rank == 0 and elapsed is not None:
            messages = 2 * iterations
            time_per_message = elapsed / messages
            if message_size == 0:
                bandwidth = 0.0
            else:
                bandwidth = (message_size / 1_000_000.0) / time_per_message

            rows.append(
                {
                    "size_bytes": str(message_size),
                    "iterations": str(iterations),
                    "total_time_seconds": f"{elapsed:.6f}",
                    "time_per_message_seconds": f"{time_per_message:.9f}",
                    "bandwidth_mb_s": f"{bandwidth:.6f}",
                }
            )
            print(
                f"{message_size:>8} bytes, {iterations:>6} iterations, "
                f"{time_per_message:.9f} s/message"
            )

    if rank == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "size_bytes",
                    "iterations",
                    "total_time_seconds",
                    "time_per_message_seconds",
                    "bandwidth_mb_s",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
