PYTHON=${PYTHON:-.venv/bin/python}

mpiexec -n 4 "$PYTHON" cellular_automata.py run --rule 30 --length 121 --steps 120 --boundary constant --initial single --output results/ca_rule30_constant.csv
mpiexec -n 4 "$PYTHON" cellular_automata.py run --rule 90 --length 121 --steps 120 --boundary constant --initial single --output results/ca_rule90_constant.csv
mpiexec -n 4 "$PYTHON" cellular_automata.py run --rule 110 --length 121 --steps 120 --boundary periodic --initial single --output results/ca_rule110_periodic.csv
