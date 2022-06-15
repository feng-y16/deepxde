#!/bin/bash
set -e
host=$1
experiments=("burgers" "convection" "navier_stokes" "schrodinger" "stiff_ode")
for experiment in "${experiments[@]}"; do
  rsync -az --include "*.png" --exclude "*.*" "$host":~/deepxde/experiments/"$experiment"/ ./experiments/"$experiment"/
  rsync -az --include "*.pdf" --exclude "*.*" "$host":~/deepxde/experiments/"$experiment"/ ./experiments/"$experiment"/
  rsync -az --include "*.txt" --exclude "*.*" "$host":~/deepxde/experiments/"$experiment"/ ./experiments/"$experiment"/
done
echo "download complete"
