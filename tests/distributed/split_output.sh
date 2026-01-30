#!/bin/bash
# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Used to split the output from each rank into an individual file in the log directory.
# export _LOGDIR=/tmp/debug
# mkdir -p ${_LOGDIR}
# torchrun --nproc-per-node 2 --no-python bash split_output.sh python3 -u my-program.py
#

if [[ "x${_LOGDIR}" == "x" ]] ; then
    echo "ERROR: _LOGDIR must be set"
    exit 1
fi

# Support non-torchrun launches
if [[ "x${RANK}" == "x" ]] ; then
  export _RANK=0
else
  export _RANK=${RANK}
fi

# Define the output log file for this process
mkdir -p ${_LOGDIR}
export _LOGFILE=output-at-rank-${_RANK}.txt

# Overwrite the file (do not append)
echo "#" > ${_LOGDIR}/${_LOGFILE}

if [[ "x$_SHOW_PROGRESS" != "x" ]] ; then
    # Rank 0 logs to stdout and to the file
    # everyone else just logs to the file
    if [[ $_RANK -eq 0 ]] ; then
        exec &> >(tee -a ${_LOGDIR}/${_LOGFILE})
    else
        exec >> ${_LOGDIR}/${_LOGFILE}
    fi
else
    exec >> ${_LOGDIR}/${_LOGFILE}
fi
exec 2>&1

#--------------------------------------------------------------
# Disable color logs
export DTLOG_COLOR=0

# Unbuffered python output
# https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUNBUFFERED
export PYTHONUNBUFFERED=1

#--------------------------------------------------------------
# Record environment
#env | sort > ${_LOGDIR}/env-at-rank-${_RANK}.txt

#--------------------------------------------------------------
echo "# ---------------------------"
echo "# "`date`
echo "# Running: "$@
echo "# ---------------------------"
$@
RTN=$?
echo "# ---------------------------"
echo "# COMPLETE: RTN=${RTN}"
echo "# ---------------------------"
exit $RTN
