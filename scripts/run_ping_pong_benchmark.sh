PYTHON=${PYTHON:-.venv/bin/python}

mpiexec -n 2 "$PYTHON" ping_pong_benchmark.py --output results/ping_pong.csv
