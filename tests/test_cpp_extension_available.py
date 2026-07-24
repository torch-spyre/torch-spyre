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

"""Gate test: fails if the C++ logging extension is not importable.

This prevents silent all-skip in C++ logging test suites.  If this test
fails, the native extension was not built or installed correctly.
"""

import subprocess
import sys
import textwrap


def test_cpp_logging_extension_is_importable():
    """The _C._logging extension must be available for C++ logging tests."""
    script = """
        import torch  # noqa: F401
        import torch_spyre  # noqa: F401
        from torch_spyre._C import _logging as cpp_logging
        config = cpp_logging.LoggingConfig.instance()
        print(f"OK components={len(config.get_components())}")
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"_C._logging is not importable — C++ logging tests will fail.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "OK" in result.stdout
