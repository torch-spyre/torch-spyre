import click
from .core import launch_from_iofile


@click.group()
def cli():
    pass


@cli.command()
@click.option("--path", default=".", help="Path to folder with SpyreCode")
@click.argument("iofile")
def launch(path, iofile):
    """Launch Spyrecode."""
    launch_from_iofile(path, iofile)


def main():
    cli()
