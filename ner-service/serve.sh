#!/usr/bin/env bash

set -xe
cd $(dirname $0)

export PYTHONUNBUFFERED=1
PYTHONIOENCODING=utf-8 ner-serving  --port=8000  2>&1 | tee ${LOG_PATH}/log

