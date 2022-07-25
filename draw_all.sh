#!/bin/bash
set -e
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
./draw.sh burgers
./draw.sh convection
./draw.sh navier_stokes
./draw.sh schrodinger
./draw.sh stiff_ode
