"""The `preflight` command. `serve --mcp` is a placeholder until its module is built."""

from pathlib import Path

from typer.testing import CliRunner

from preflight.cli import app

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "scenes" / "kitchen_rules.yaml"

runner = CliRunner()


def test_help_lists_the_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "replay" in result.output and "serve" in result.output


def test_replay_needs_an_id(tmp_path):
    log = tmp_path / "audit.jsonl"
    log.touch()
    assert runner.invoke(app, ["replay", str(log)]).exit_code == 2


def test_replay_needs_an_existing_log(tmp_path):
    assert runner.invoke(app, ["replay", str(tmp_path / "missing.jsonl"), "--id", "1"]).exit_code == 2


def test_replay_of_an_empty_log_fails(tmp_path):
    log = tmp_path / "audit.jsonl"
    log.touch()
    result = runner.invoke(app, ["replay", str(log), "--id", "1"])
    assert result.exit_code == 1
    assert "no record with id 1" in result.output


def test_serve_needs_mcp():
    result = runner.invoke(app, ["serve", "--rules", str(RULES)])
    assert result.exit_code == 2
    assert "give --mcp" in result.output


def test_serve_refuses_an_unknown_feedback_mode():
    assert runner.invoke(app, ["serve", "--mcp", "--rules", str(RULES), "--feedback", "loud"]).exit_code == 2


def test_serve_mcp_is_not_built_yet():
    result = runner.invoke(app, ["serve", "--mcp", "--rules", str(RULES), "--feedback", "minimal"])
    assert result.exit_code == 1
    assert "preflight serve --mcp is not built yet" in result.output
