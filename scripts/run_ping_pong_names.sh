PYTHON=${PYTHON:-.venv/bin/python}

mpiexec -n 4 "$PYTHON" ping_pong_names.py --passes 12 --seed 7
