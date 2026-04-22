# content of conftest.py
import os


def pytest_addoption(parser):
    parser.addoption(
        "--cpu-compile",
        action="store_true",
        default=False,
        help="enable cpu compile comparisons",
    )


def pytest_configure(config):
    compile_enabled = config.getoption("--cpu-compile", default=False)
    # note that we don't set the env variable if not True
    if compile_enabled:
        os.environ["TEST_COMPARE_CPU_COMPILE"] = str(True)
