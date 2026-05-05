PYTHON=${PYTHON:-.venv/bin/python}

TEMP_FILES="results/ca_speedup_p1.csv results/ca_speedup_p2.csv results/ca_speedup_p4.csv results/ca_speedup_p8.csv"

rm -f $TEMP_FILES
rm -f results/ca_speedup_raw.csv results/ca_speedup.csv

for PROCESSES in 1 2 4 8
do
    mpiexec -n "$PROCESSES" "$PYTHON" cellular_automata.py benchmark --rule 110 --length 20000 --steps 300 --boundary periodic --initial random --seed 7 --output "results/ca_speedup_p${PROCESSES}.csv"
done

"$PYTHON" scripts/build_ca_speedup.py

rm -f $TEMP_FILES
