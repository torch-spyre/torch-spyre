#! /bin/bash
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

usage()
{
    echo "Usage: $(basename $0) > metric.bin"
    echo ""
    echo "Create a metric file in the new format for testing"
    echo ""
    echo "Env vars:"
    echo "   DEBUG:  Set '1' or 'true' or 'yes' to turn on debug mode."
    echo "   ENDIAN: Set 'BE' or 'LE' to force endian."
    exit 0
}

[[ x"$1" = x'-h' || x"$1" = x'--help' ]] && usage

if [[ -z "$ENDIAN" ]]; then
    case $(uname -m) in
    s390x)  ENDIAN='BE';;
    *)      ENDIAN='LE';;
    esac
fi

DEBUG=${DEBUG:-"no"}

MAGIC=('"A"' '"i"' '"u"' '"M"')

is_debug()
{
    case "$DEBUG" in
    0|[nN]*|[fF]*)  return 1;;
    *)              return 0;;
    esac
}

log_debug()
{
    if is_debug; then
        echo $*
    fi
}

gen()
{
    case $1 in
    8)   sz=1;;
    16)  sz=2;;
    32)  sz=4;;
    64)  sz=8;;
    *)     echo "ERROR: Unexpected size: $1" >&2
           return;;
    esac

    if [[ $# -ge 2 ]]; then
        endian=$2
    else
        endian=$ENDIAN
    fi

    if is_debug; then
        cat
    else
        if [[ $endian = 'LE' ]]; then
            xxd -r -p | xxd -e -g $sz | xxd -r
        else
            xxd -r -p
        fi
    fi
}

gen_file_header()
{
    size=$1
    nsec=$2
    (   printf '%02x %02x %02x %02x\n' ${MAGIC[@]}  # Matic
    ) | gen 8
    (   printf '%08x\n' $size                       # File size
    ) | gen 32
    (   echo '01 00 00 01'                          # Version
    ) | gen 8
    (   printf '%08x\n' $nsec                       # Num sections
    ) | gen 32
    (   printf '%016x\n' $(( $(date +%s) * 1000 ))  # Last update
        echo '0000000000000000'                     # reserved
    ) | gen 64
}

gen_section_header()
{
    type=$1
    size=$2
    ndat=$3
    (   printf '%08x\n' $type        # Section type
        printf '%08x\n' $size        # Section size
        printf '%08x\n' $(( 1$(date +%N) - 1000000000 ))  # Sequence number
        printf '%08x\n' $ndat        # Num data arrays
    ) | gen 32
    (   printf '%016x\n' $(( $(date +%s) * 1000 ))  # Last update
    ) | gen 64
    (   printf '%08x\n' 100          # Interval
        echo '00000000'              # reserved
    ) | gen 32
}

gen_type_size_table()
{
    while [[ $# -ge 2 ]]
    do
        type=$1
        size=$2
        shift 2
        printf '%08x %08x\n' $type $size
    done | gen 32

    [[ $# -eq 1 ]] && echo "ERROR: Number of table parameters is an odd number" >&2
}

gen_data()
{
    for d in $@
    do
        printf '%016x\n' $d
    done | gen 64
}

log_debug "=== File header"
gen_file_header 216 2
gen_type_size_table 16 64 32 104

log_debug "--- Section header #1"
gen_section_header 16 64 2
gen_type_size_table 129 1 130 1
log_debug " -- Section data #1"
gen_data 1024 2048

log_debug "--- Section header #2"
gen_section_header 32 104 3
gen_type_size_table 16 1 18 2 19 3
log_debug " -- Section data #2"
gen_data 420
gen_data 65536 64
gen_data 49152 32 8
