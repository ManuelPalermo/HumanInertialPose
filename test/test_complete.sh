#!/bin/bash

# same as test.sh but also checks docstrings and pep8
CURRENT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )";
PARENT_DIR="$(dirname "$CURRENT_DIR")";
python -m pytest \
       --flake8 --pydocstyle \
       --cov=$PARENT_DIR/hipose \
       --cov-report=term-missing \
       --cov-report=xml:./coverage_report.xml
