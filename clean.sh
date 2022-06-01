#!/bin/bash
exp_name=$1
rm -rf experiments/"$exp_name"/*_info.pkl
rm -rf experiments/"$exp_name"/*.png
rm -rf experiments/"$exp_name"/*.pdf
rm -rf experiments/"$exp_name"/*.txt
rm -rf experiments/"$exp_name"/*.log
