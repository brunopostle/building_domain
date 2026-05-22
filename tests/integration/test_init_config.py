"""Integration tests for bsos init and bsos config commands."""
import os
from pathlib import Path
from typer.testing import CliRunner
from bsos.cli.main import app

runner = CliRunner()


def test_init_creates_database(tmp_path):
    db = tmp_path / "test.db"
    result = runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    assert result.exit_code == 0, result.output
    assert db.exists()
    assert "Done." in result.output


def test_init_writes_bsos_config(tmp_path):
    db = tmp_path / "test.db"
    config_file = tmp_path / ".bsos_config"
    orig_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
        assert result.exit_code == 0, result.output
        assert config_file.exists()
        assert str(db) in config_file.read_text()
    finally:
        os.chdir(orig_dir)


def test_init_appends_to_gitignore(tmp_path):
    db = tmp_path / "test.db"
    orig_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--db", str(db)])
        assert result.exit_code == 0, result.output
        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()
        assert ".bsos_config" in gitignore.read_text()
    finally:
        os.chdir(orig_dir)


def test_init_fails_if_db_exists(tmp_path):
    db = tmp_path / "test.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    result = runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_init_force_on_existing_db(tmp_path):
    db = tmp_path / "test.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    result = runner.invoke(app, ["init", "--db", str(db), "--force", "--no-gitignore"])
    assert result.exit_code == 0, result.output
    assert "up to date" in result.output


def test_init_writes_default_config_keys(tmp_path):
    db = tmp_path / "test.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    result = runner.invoke(app, ["config", "list", "--db", str(db)])
    assert result.exit_code == 0
    assert "graph_rebuild_threshold" in result.output
    assert "embedding_model" in result.output


def test_config_set_and_get(tmp_path):
    db = tmp_path / "test.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    runner.invoke(app, ["config", "set", "default_llm_model", "gpt-4o", "--db", str(db)])
    result = runner.invoke(app, ["config", "get", "default_llm_model", "--db", str(db)])
    assert result.exit_code == 0
    assert "gpt-4o" in result.output


def test_config_get_unset_key(tmp_path):
    db = tmp_path / "test.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    result = runner.invoke(app, ["config", "get", "nonexistent_key", "--db", str(db)])
    assert result.exit_code == 0
    assert "(not set)" in result.output


def test_config_unset(tmp_path):
    db = tmp_path / "test.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    runner.invoke(app, ["config", "set", "default_llm_model", "gpt-4o", "--db", str(db)])
    runner.invoke(app, ["config", "unset", "default_llm_model", "--db", str(db)])
    result = runner.invoke(app, ["config", "get", "default_llm_model", "--db", str(db)])
    assert "(not set)" in result.output


def test_config_unknown_key_warns(tmp_path):
    db = tmp_path / "test.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    result = runner.invoke(app, ["config", "set", "totally_unknown_key", "val", "--db", str(db)])
    assert result.exit_code == 0
    assert "Unknown" in result.output
