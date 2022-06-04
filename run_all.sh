#!/bin/bash
set -e
bash run.sh burgers &
bash run.sh convection &
bash run.sh navier_stokes &
bash run.sh stiff_ode &
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$num_jobs" "jobs remaining"
  sleep 20
done
bash run.sh schrodinger &
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$num_jobs" "jobs remaining"
  sleep 20
done
echo "all experiments complete"
