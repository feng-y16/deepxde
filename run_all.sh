#!/bin/bash
set -e
bash run.sh burgers &
sleep 10
bash run.sh convection &
sleep 10
bash run.sh navier_stokes &
sleep 10
bash run.sh stiff_ode &
num_jobs=$(jobs | grep -c "")
set +e
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$num_jobs" "jobs remaining"
  sleep 20
done
set -e
bash run.sh schrodinger &
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$num_jobs" "jobs remaining"
  sleep 20
done
echo "all experiments complete"
