import argparse
import random
from mpi4py import MPI


def choose_next(rng, rank, size):
    # pick a random node that isn't us
    choice = rng.randrange(size - 1)
    if choice >= rank:
        choice += 1
    return choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # offset seed by rank so everyone gets different random numbers
    rng = random.Random(args.seed + rank)

    if rank == 0:
        target = choose_next(rng, rank, size)
        msg = {"path": [0, target], "remaining": args.passes - 1}
        print(f"0 -> {target}")
        comm.ssend(msg, dest=target, tag=0)

    while True:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        if msg.get("stop"):
            break

        path = msg["path"]
        rem = msg["remaining"]
        
        # print("DEBUG: got msg", path)
        print(" -> ".join(str(item) for item in path))

        if rem == 0:
            print(f"Finished after {len(path) - 1} passes.")
            # tell everyone else to stop
            for other in range(size):
                if other != rank:
                    comm.send({"stop": True}, dest=other, tag=1)
            break

        next_rank = choose_next(rng, rank, size)
        next_msg = {"path": path + [next_rank], "remaining": rem - 1}
        comm.ssend(next_msg, dest=next_rank, tag=0)


if __name__ == "__main__":
    main()
