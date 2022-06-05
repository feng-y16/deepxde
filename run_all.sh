#!/bin/bash
set -e
bash run.sh burgers &
sleep 10
bash run.sh convection &
sleep 10
bash run.sh navier_stokes &
sleep 10
bash run.sh stiff_ode &
sleep 10
bash run.sh schrodinger &
set +e
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$num_jobs" "jobs remaining"
  sleep 20
done
set -e
echo "all experiments complete"
