#!/bin/bash
set -e
match_phrase=$1
pgrep -u fengyao "$match_phrase" | awk "{print \"kill \"\$1}" | sh
