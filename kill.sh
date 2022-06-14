#!/bin/bash
set -e
if [ $# -gt 0 ]; then
  match_phrase=$1
else
  match_phrase="python"
fi
pgrep -u ${USER} "$match_phrase" | awk "{print \"kill \"\$1}" | sh
