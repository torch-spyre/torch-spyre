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

"""
Example usage of the spyremetrics package.

This script demonstrates how to read and iterate through a binary metric file
using spyremetrics APIs.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from traceback import print_tb, print_exception, print_exc

from spyremetrics import MetricFile

USAGE = """\
Usage: python example.py <metric_file_path> [<option>]

Option:
   Integer-value: Interval to repeat printing single-line output
   no, false, 0:  Show all data in a metric file. Do not add host_metrics pseudo section.
   Otherwise:     Show all data in a metric file. Add host_metrics pseudo section
                  if host_metrics are not stored in a metric file.
"""


def main() -> None:
    """Main function to demonstrate metric file parsing."""
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    if sys.argv[1] in ("-h", "--help", "-?"):
        print(USAGE)
        sys.exit(0)

    interval = 0
    show_detail = False
    host_metrics = False

    if len(sys.argv) >= 3 and bool(sys.argv[2]):
        try:
            interval = int(sys.argv[2])  ## when 2nd arg is an integer
        except ValueError:
            ## when 2nd arg is not an integer
            show_detail = True
            host_metrics = not (sys.argv[2].lower().startswith(("n", "f", "0", "-")))

    # Expand "%BUSID" magic keyword if needed and open metric files. Return a list of MetricFile objects
    metric_file_list = MetricFile.open_metrics(
        sys.argv[1], local_host_metrics=host_metrics
    )

    try:
        if show_detail:
            for i, mf in enumerate(metric_file_list):
                #                print('-------- Debug dump of MetricFile --------', file=sys.stderr)
                #                print(str(mf), file=sys.stderr)
                #                print('------------------------------------------', file=sys.stderr)

                print(f"Metric file #{i}: {mf.path}")
                print(f"Number of sections: {mf.nsections}")
                print()

                # Iterate through all sections stored in the file and dump all metrics
                for section_idx, section in enumerate(mf.sections()):
                    print(f"Section {section_idx}:")
                    print(f"  Type: {section.type_id} ({section.type.name})")
                    print(f"  Data word count: {section.len}")

                    # Iterate through all metric data in the file
                    print("  Metrics:")
                    for i, (metric, val) in enumerate(section.items()):
                        print(f"    [{i:#2d}]: {hex(int(val))} ({val}) {str(metric)}")

                    print()
        else:
            for i, mf in enumerate(metric_file_list):
                # Select metrics we are interested in.
                # MetricsFile.read_metrics() only returns the metrics listed in the argument.
                # Default is to return all metrics available in the metric file.
                # If the metric is not stored in the file, it is not included in the retrued data, of cource.
                mf.set_filters(["pwr", "tempr", "rdmem", "wrmem", "avgmem", "peakmem"])
                print(f"Metric file #{i}: {mf.path} ({mf.nsections} sections)")
            print()
            print("ID  Timestamp           pwr   tempr rdmem  wrmem  avgmem peakmem")
            while True:
                for i, mf in enumerate(metric_file_list):
                    pwr = tempr = rdmem = wrmem = avgmem = peakmem = 0.0
                    for metric, val in mf.read_metrics():
                        # pick up interested metrics and adjust metric values to be human friendly
                        match metric.name:
                            case "pwr":
                                pwr = val
                            case "tempr":
                                tempr = val
                            case "rdmem":
                                rdmem = val * 1.0 / 1024 / 1024  # convert to GiB
                            case "wrmem":
                                wrmem = val * 1.0 / 1024 / 1024  # convert to GiB
                            case "avgmem":
                                avgmem = val * 1.0 / 1024 / 1024  # convert to GiB
                            case "peakmem":
                                peakmem = val * 1.0 / 1024 / 1024  # convert to GiB

                    print(
                        f"{i:>2}: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} "
                        f"{pwr:5.1f} {tempr:5.1f} {rdmem:6.1f} {wrmem:6.1f} {avgmem:6.1f} {peakmem:6.1f}"
                    )

                if interval < 1:
                    break
                else:
                    time.sleep(interval)

    except RuntimeError as e:
        print(f"Error while handling a metric file: {e}", file=sys.stderr)
        #        print_exception(e, file=sys.stderr)
        print_exc(file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        #        print_exception(e, file=sys.stderr)
        print_exc(file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
