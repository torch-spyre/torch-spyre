import argparse
import glob
import os
import pytest


def main():
    parser = argparse.ArgumentParser(description="driver for running test cases derived from models") 
    parser.add_argument("--opname", help="torch operation name to be tested")
    parser.add_argument("--ignore_skip_files", action="store_true", help="ignore model/skip_files.yaml")
    args = parser.parse_args()

    if args.ignore_skip_files:
        os.environ["TEST_MODELS_OPS_IGNORE_SKIP_FILES"] = "1"

    if args.opname is None:
        targets = ["models"]
    else:
        targets = glob.glob(f"models/*/test_{args.opname}_*.py")

    pytest.main(targets)

if __name__ == "__main__":
    main()
