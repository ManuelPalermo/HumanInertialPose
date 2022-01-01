#!/bin/bash

CURRENT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )";
PARENT_DIR="$(dirname "$CURRENT_DIR")";
python -m pytest \
       --cov=$PARENT_DIR/hipose \
       --cov-report=term-missing \
       --cov-report=xml:./coverage_report.xml
