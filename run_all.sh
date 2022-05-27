#!/bin/bash
bash run.sh burgers &
bash run.sh navier_stokes &
bash run.sh stiff_ode &
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  jobs
  num_jobs=$(jobs | grep -c "")
  sleep 60
done
bash run_sensitivity.sh
echo "all experiments complete"
