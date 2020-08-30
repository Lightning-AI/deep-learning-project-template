from unittest import mock

import pytest


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cpu_template(cli_args):
    """Test running CLI for an example with default params."""
    from research_mnist.mnist_trainer import main_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        main_cli()
