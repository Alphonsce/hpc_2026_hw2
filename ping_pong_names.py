import argparse
import random

from mpi4py import MPI


def choose_next(rng, rank, size):
    choice = rng.randrange(size - 1)
    if choice >= rank:
        choice += 1
    return choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def send_stop(comm, rank, size):
    for other_rank in range(size):
        if other_rank != rank:
            comm.send({"stop": True}, dest=other_rank, tag=1)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rng = random.Random(args.seed + rank)

    if rank == 0:
        first_target = choose_next(rng, rank, size)
        first_message = {"path": [0, first_target], "remaining": args.passes - 1}
        print(f"0 -> {first_target}")
        comm.ssend(first_message, dest=first_target, tag=0)

    while True:
        message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        if message.get("stop"):
            break

        path = message["path"]
        remaining = message["remaining"]
        print(" -> ".join(str(item) for item in path))

        if remaining == 0:
            print(f"Finished after {len(path) - 1} passes.")
            send_stop(comm, rank, size)
            break

        next_rank = choose_next(rng, rank, size)
        next_message = {"path": path + [next_rank], "remaining": remaining - 1}
        comm.ssend(next_message, dest=next_rank, tag=0)


if __name__ == "__main__":
    main()
