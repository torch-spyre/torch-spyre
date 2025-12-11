# Copyright 2025 The Torch-Spyre Authors.
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

import shutil
from pathlib import Path
import os
from typing import cast

os.environ.setdefault(
    "TORCH_DEVICE_BACKEND_AUTOLOAD", "0"
)  # must be before torch import

import glob


from setuptools import setup, Command
from torch.utils.cpp_extension import BuildExtension, CppExtension

PATH_NAME = "torch_spyre"
PACKAGE_NAME = "torch_spyre"


def get_torch_spyre_version() -> str:
    version_ns: dict[str, object] = {}
    with open(f"{PATH_NAME}/version.py") as f:
        exec(f.read(), version_ns)
        version = cast(str, version_ns["__version__"])
    return version


version = get_torch_spyre_version()


def check_libflex():
    ld_library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    for path in ld_library_paths:
        if glob.glob(os.path.join(path, "libflex.so")):
            return True
    return False


ROOT_DIR = Path(__file__).absolute().parent
CODEGEN_DIR = ROOT_DIR / "codegen"
CSRC_DIR = ROOT_DIR / PATH_NAME / "csrc"


# Automatically download json.hpp if not present
def maybe_download_nlohmann_json():
    """return path to header files"""
    import urllib.request

    NLOHMANN_URL = "https://raw.githubusercontent.com/nlohmann/json/v3.11.2/single_include/nlohmann/json.hpp"
    SHARED_PATH = Path(
        os.environ.get("SHARED_DEPS_DIR", ROOT_DIR / PATH_NAME / "csrc" / "external")
    )
    NLOHMANN_INC_DIR = SHARED_PATH / "nlohmann" / "include"
    NLOHMANN_DIR = NLOHMANN_INC_DIR / "nlohmann"

    NLOHMANN_HEADER = os.path.join(NLOHMANN_DIR, "json.hpp")
    if not os.path.exists(NLOHMANN_HEADER):
        os.makedirs(NLOHMANN_DIR, exist_ok=True)
        print("Downloading nlohmann/json.hpp...")
        urllib.request.urlretrieve(NLOHMANN_URL, NLOHMANN_HEADER)
    return NLOHMANN_INC_DIR


INCLUDE_DIRS = [
    CSRC_DIR,
    # "tracy/public"
]
LIBRARY_DIRS = []


INCLUDE_DIRS += [maybe_download_nlohmann_json()]

cmake_include_path = os.environ.get("CMAKE_INCLUDE_PATH", "")
extra_include_dirs = cmake_include_path.split(":") if cmake_include_path else []
INCLUDE_DIRS += [Path(p) for p in extra_include_dirs if p]

cmake_library_path = os.environ.get("CMAKE_LIBRARY_PATH", "")
extra_library_dirs = cmake_library_path.split(":") if cmake_library_path else []
LIBRARY_DIRS += [Path(p) for p in extra_library_dirs if p]

if "RUNTIME_INSTALL_DIR" in os.environ:
    # take lower precedence than CMAKE_LIBRARY_PATH and CMAKE_INCLUDE_PATH
    RUNTIME_DIR = Path(os.environ["RUNTIME_INSTALL_DIR"])
    SENLIB_DIR = Path(os.environ["SENLIB_INSTALL_DIR"])
    DEEPTOOLS_DIR = Path(os.environ["DEEPTOOLS_INSTALL_DIR"])
    INCLUDE_DIRS += [
        RUNTIME_DIR / "include",
    ]
    INCLUDE_DIRS += [
        RUNTIME_DIR / "include" / "concurrentqueue" / "moodycamel",
    ]
    INCLUDE_DIRS += [
        SENLIB_DIR / "include",
    ]
    INCLUDE_DIRS += [
        DEEPTOOLS_DIR / "include",
    ]
    LIBRARY_DIRS += [RUNTIME_DIR / "lib"]

LIBRARIES = ["sendnn", "flex", "dee_internal"]

# FIXME: added no-deprecated as this fails in sentensor_shape.hpp
# - we need to fix there
# Note that we always compile with debug info
# EXTRA_CXX_FLAGS = ["-g", "-Wall", "-Werror", "-Wno-deprecated"]
EXTRA_CXX_FLAGS = ["-g", "-Wall", "-Wno-deprecated", "-std=c++17"]


class clean(Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove torch_spyre extension
        for path in (ROOT_DIR / PATH_NAME).glob("**/*.so"):
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def run_codegen():
    import sys
    import importlib

    is_meta = any(
        cmd in sys.argv for cmd in ["dist_info", "egg_info", "install_egg_info"]
    )

    if not importlib.util.find_spec("sendnn"):
        if not is_meta:
            raise ImportError("sendnn is required for building. Install it first.")

        print("Skipping codegen (sendnn not available, metadata extraction only)")
        return None

    gen_script = CODEGEN_DIR / "gen.py"

    if not gen_script.exists():
        raise FileNotFoundError(f"Codegen script not found: {gen_script}")

    print("Running codegen...")
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen", gen_script)
    assert spec is not None
    assert spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen.generate_and_register_wrappers(CODEGEN_DIR)


if __name__ == "__main__":
    OUTPUT_CODEGEN_DIR = run_codegen()

    sources = list(CSRC_DIR.glob("*.cpp"))
    if OUTPUT_CODEGEN_DIR:
        sources += list(OUTPUT_CODEGEN_DIR.glob("*.cpp"))
    sources = [str(p.relative_to(ROOT_DIR).as_posix()) for p in sorted(sources)]
    sources = sorted([str(s) for s in sources])

    ext_modules = [
        CppExtension(
            name=f"{PACKAGE_NAME}._C",
            sources=sources,
            include_dirs=[str(p) for p in INCLUDE_DIRS],
            library_dirs=[str(p) for p in LIBRARY_DIRS],
            libraries=LIBRARIES,
            extra_compile_args={"cxx": EXTRA_CXX_FLAGS},
            define_macros=[
                ("PACKAGE_NAME", f'"{PACKAGE_NAME}"'),
                ("MODULE_NAME", f'"{PACKAGE_NAME}._C"'),
                ("SPYRE_DEBUG_ENV", '"TORCH_SPYRE_DEBUG"'),
                ("EAGER_MODE_ENV", '"EAGER_MODE"'),
                ("BOOST_ALL_DYN_LINK", None),  # avoid static link to boost
            ],
        )
    ]

    setup(
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": BuildExtension.with_options(
                no_python_abi_suffix=True, verbose=True
            ),
            "clean": clean,
        },
        entry_points={
            "torch.backends": [
                "torch_spyre = torch_spyre:_autoload",
            ],
        },
    )
