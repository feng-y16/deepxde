#!/bin/bash
set -e
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
./run.sh burgers &
./run.sh stiff_ode &
./run.sh navier_stokes &
set +e
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$num_jobs" "jobs remaining"
  sleep 20
done
set -e
./run.sh schrodinger &
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
