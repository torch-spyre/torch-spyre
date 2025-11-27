import argparse
import glob
import os
import pytest


def main():
    parser = argparse.ArgumentParser(description="driver for running test cases derived from models") 
    parser.add_argument("--opname", help="torch operation name to be tested")
    parser.add_argument("--ignore_skip_files", action="store_true", help="ignore model/skip_files.yaml")
    parser.add_argument("--report", type=str, help="Show extra test summary info as specified by chars: (f)ailed, (E)rror, (s)kipped, (x)failed, (X)passed, (p)assed, (P)assed with output, (a)ll except passed (p/P), or (A)ll. (w)arnings are enabled by default (see --disable-warnings), 'N' can be used to reset the list. (default: 'fE').")
    args = parser.parse_args()

    targets = []
    if args.ignore_skip_files:
        os.environ["TEST_MODELS_OPS_IGNORE_SKIP_FILES"] = "1"

    if args.opname is None:
        targets = ["models"]
    else:
        targets = glob.glob(f"models/*/test_{args.opname}_*.py")

    if args.report:
        targets.append("-r" + args.report)

    pytest.main(targets)

if __name__ == "__main__":
    main()
