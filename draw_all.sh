#!/bin/bash
set -e
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
./draw.sh stiff_ode
./draw.sh burgers
./draw.sh navier_stokes
./draw.sh schrodinger
./draw_sensitivity.sh burgers
./draw_sensitivity.sh schrodinger
