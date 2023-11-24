# -*- coding: utf-8 -*-

"""Console script for intake_nwp."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for intake_nwp."""
    click.echo("Replace this message by putting your code into "
               "intake_nwp.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
