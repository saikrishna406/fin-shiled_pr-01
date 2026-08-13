from click.testing import CliRunner
from finshield.cli.main import main

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Fin-Shield Analytics" in result.output

def test_cli_subcommands():
    runner = CliRunner()
    commands = [
        "generate-data",
        "preprocess",
        "train",
        "evaluate",
        "infer",
        "simulate",
        "ids",
        "liquidity",
        "settle",
        "blockchain",
        "experiment",
        "report",
    ]
    for cmd in commands:
        result = runner.invoke(main, [cmd])
        assert result.exit_code == 0
