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

import click
from .core import launch_from_cli


@click.group()
def cli():
    pass


@cli.command()
@click.option("-i", "--input", multiple=True, help="input tensor")
@click.option("-o", "--output", multiple=True, help="output tensor")
@click.option(
    "--bind",
    multiple=True,
    metavar="SYM=N",
    help="bind a symbolic dimension, e.g. --bind s0=128",
)
@click.argument("path", default=".")
def launch(path, input, output, bind):
    """Launch Spyrecode.

    With no -i/-o, the tensors are built from the folder's launch_spec.json.
    Given explicitly, they are checked against it when one is present.
    """

    input = list(input)
    output = list(output)

    bindings = {}
    for item in bind:
        if "=" not in item:
            raise ValueError(f"Expected --bind SYM=N, got: {item}")
        sym, _, value = item.partition("=")
        bindings[sym] = int(value)

    # An output is required only when describing the tensors by hand: a spec
    # already says which arguments are outputs.
    if output == [] and input != []:
        raise ValueError("At least 1 output tensor expected!")

    launch_from_cli(path, input, output, bindings)


def main():
    cli()
