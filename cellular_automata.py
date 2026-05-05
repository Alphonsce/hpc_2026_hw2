import argparse
import csv
import random
import os
from pathlib import Path

from mpi4py import MPI


def parse_rule(spec):
    spec = spec.strip()

    if spec.isdigit():
        rule_num = int(spec)

        def wolfram_rule(left, center, right):
            # magic bitwise stuff to get the wolfram rule output
            idx = (left << 2) | (center << 1) | right
            return (rule_num >> idx) & 1

        return wolfram_rule

    # explicit mapping
    mapping = {}
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        pattern, val = item.split(":", 1)
        mapping[pattern.strip()] = int(val.strip())

    def explicit_rule(left, center, right):
        return mapping[f"{left}{center}{right}"]

    return explicit_rule


def flatten(chunks):
    res = []
    for c in chunks:
        res.extend(c)
    return res


def exchange_ghosts(comm, local_state, boundary, const_val):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        if boundary == "periodic":
            return local_state[-1], local_state[0]
        return const_val, const_val

    if boundary == "periodic":
        left_r = (rank - 1) % size
        right_r = (rank + 1) % size
    else:
        left_r = rank - 1 if rank > 0 else MPI.PROC_NULL
        right_r = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # send left, recv right
    ghost_l = comm.sendrecv(
        sendobj=local_state[-1],
        dest=right_r,
        sendtag=10,
        source=left_r,
        recvtag=10,
    )
    
    # send right, recv left
    ghost_r = comm.sendrecv(
        sendobj=local_state[0],
        dest=left_r,
        sendtag=11,
        source=right_r,
        recvtag=11,
    )

    if ghost_l is None:
        ghost_l = const_val
    if ghost_r is None:
        ghost_r = const_val
        
    return ghost_l, ghost_r


def run_steps(
    comm,
    length,
    steps,
    rule,
    boundary,
    initial_mode,
    seed,
    const_val,
    collect_hist,
):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if initial_mode == "single":
            init_state = [0] * length
            init_state[length // 2] = 1
        elif initial_mode == "alternating":
            init_state = [i % 2 for i in range(length)]
        else:
            rng = random.Random(seed)
            init_state = [rng.randint(0, 1) for _ in range(length)]

        # split data for scatter
        chunks = []
        start = 0
        base, extra = divmod(length, size)
        for i in range(size):
            c_size = base + (1 if i < extra else 0)
            chunks.append(init_state[start : start + c_size])
            start += c_size
    else:
        chunks = None

    local_state = comm.scatter(chunks, root=0)
    history = [] if (rank == 0 and collect_hist) else None

    if collect_hist:
        gathered = comm.gather(local_state, root=0)
        if rank == 0:
            history.append(flatten(gathered))

    comm.Barrier()
    t0 = MPI.Wtime()

    for _ in range(steps):
        ghost_l, ghost_r = exchange_ghosts(
            comm, local_state, boundary, const_val
        )
        
        next_state = []
        for i, val in enumerate(local_state):
            l_val = ghost_l if i == 0 else local_state[i - 1]
            r_val = ghost_r if i == len(local_state) - 1 else local_state[i + 1]
            next_state.append(rule(l_val, val, r_val))
            
        local_state = next_state

        if collect_hist:
            gathered = comm.gather(local_state, root=0)
            if rank == 0:
                history.append(flatten(gathered))

    local_time = MPI.Wtime() - t0
    elapsed = comm.reduce(local_time, op=MPI.MAX, root=0)
    
    return history, elapsed


def main():
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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rule = parse_rule(args.rule)

    collect_hist = args.command == "run"
    history, elapsed = run_steps(
        comm,
        args.length,
        args.steps,
        rule,
        args.boundary,
        args.initial,
        args.seed,
        args.constant_value,
        collect_hist,
    )

    if rank != 0 or elapsed is None:
        return

    if args.command == "run":
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(history or [])
        print(f"Saved {args.output}")
    else:
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["processes", "time_seconds"])
                writer.writeheader()
                writer.writerow({
                    "processes": str(comm.Get_size()),
                    "time_seconds": f"{elapsed:.6f}",
                })
        print(f"{comm.Get_size()} processes: {elapsed:.6f} s")


if __name__ == "__main__":
    main()
