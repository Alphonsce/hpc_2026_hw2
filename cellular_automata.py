import argparse
import csv
import random
from pathlib import Path

from mpi4py import MPI


def split_array(values, parts):
    chunks = []
    start = 0
    base, extra = divmod(len(values), parts)
    for index in range(parts):
        chunk_size = base + (1 if index < extra else 0)
        chunks.append(values[start : start + chunk_size])
        start += chunk_size
    return chunks


def make_initial_state(length, mode, seed):
    if mode == "single":
        state = [0] * length
        state[length // 2] = 1
        return state
    if mode == "alternating":
        return [index % 2 for index in range(length)]
    rng = random.Random(seed)
    return [rng.randint(0, 1) for _ in range(length)]


def parse_rule(spec):
    spec = spec.strip()

    if spec.isdigit():
        rule_number = int(spec)

        def wolfram_rule(left, center, right):
            index = (left << 2) | (center << 1) | right
            return (rule_number >> index) & 1

        return wolfram_rule

    mapping = {}
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        pattern, value = item.split(":", 1)
        mapping[pattern.strip()] = int(value.strip())

    def explicit_rule(left, center, right):
        return mapping[f"{left}{center}{right}"]

    return explicit_rule


def flatten_chunks(chunks):
    flattened = []
    for chunk in chunks:
        flattened.extend(chunk)
    return flattened


def exchange_ghosts(comm, local_state, boundary, constant_value):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        if boundary == "periodic":
            return local_state[-1], local_state[0]
        return constant_value, constant_value

    if boundary == "periodic":
        left_rank = (rank - 1) % size
        right_rank = (rank + 1) % size
    else:
        left_rank = rank - 1 if rank > 0 else MPI.PROC_NULL
        right_rank = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    left_ghost = comm.sendrecv(
        sendobj=local_state[-1], dest=right_rank, sendtag=10, source=left_rank, recvtag=10
    )
    right_ghost = comm.sendrecv(
        sendobj=local_state[0], dest=left_rank, sendtag=11, source=right_rank, recvtag=11
    )

    if left_ghost is None:
        left_ghost = constant_value
    if right_ghost is None:
        right_ghost = constant_value
    return left_ghost, right_ghost


def run_steps(comm, length, steps, rule, boundary, initial_mode, seed, constant_value, collect_history):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        initial_state = make_initial_state(length, initial_mode, seed)
        chunks = split_array(initial_state, size)
    else:
        chunks = None

    local_state = comm.scatter(chunks, root=0)
    history = [] if (rank == 0 and collect_history) else None

    if collect_history:
        gathered = comm.gather(local_state, root=0)
        if rank == 0:
            history.append(flatten_chunks(gathered))

    comm.Barrier()
    start_time = MPI.Wtime()

    for _ in range(steps):
        left_ghost, right_ghost = exchange_ghosts(comm, local_state, boundary, constant_value)
        next_state = []
        for index, value in enumerate(local_state):
            left_value = left_ghost if index == 0 else local_state[index - 1]
            right_value = right_ghost if index == len(local_state) - 1 else local_state[index + 1]
            next_state.append(rule(left_value, value, right_value))
        local_state = next_state

        if collect_history:
            gathered = comm.gather(local_state, root=0)
            if rank == 0:
                history.append(flatten_chunks(gathered))

    local_elapsed = MPI.Wtime() - start_time
    elapsed = comm.reduce(local_elapsed, op=MPI.MAX, root=0)
    return history, elapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "benchmark"])
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--rule", default="30")
    parser.add_argument("--boundary", choices=["periodic", "constant"], default="constant")
    parser.add_argument("--initial", choices=["single", "alternating", "random"], default="single")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--constant-value", type=int, choices=[0, 1], default=0)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rule = parse_rule(args.rule)

    collect_history = args.command == "run"
    history, elapsed = run_steps(
        comm, args.length, args.steps, rule, args.boundary,
        args.initial, args.seed, args.constant_value, collect_history
    )

    if rank != 0 or elapsed is None:
        return

    if args.command == "run":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(history or [])
        print(f"Saved {args.output}")
    else:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["processes", "time_seconds"])
                writer.writeheader()
                writer.writerow({"processes": str(comm.Get_size()), "time_seconds": f"{elapsed:.6f}"})
                
        print(f"{comm.Get_size()} processes: {elapsed:.6f} s")


if __name__ == "__main__":
    main()
