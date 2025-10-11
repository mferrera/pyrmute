"""Test that all examples run without errors."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_FILES = [
    "advanced_features.py",
    "config_file_migration.py",
    "etl_data_import.py",
    "message_queue_consumer.py",
    "ml_inference_pipeline.py",
]


@pytest.fixture
def examples_dir(source_root: Path) -> Path:
    """Get the examples directory."""
    return source_root / "examples"


@pytest.mark.examples
@pytest.mark.parametrize("example_file", EXAMPLE_FILES)
def test_example_runs(
    example_file: str,
    examples_dir: Path,
    tmp_path: Path,
) -> None:
    """Test that example runs without errors in isolated temp directory."""
    example_path = examples_dir / example_file

    if not example_path.exists():
        pytest.skip(f"Example file not found: {example_path}")

    result = subprocess.run(
        [sys.executable, str(example_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )

    assert result.returncode == 0, (
        f"Example {example_file} failed:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "=" in result.stdout, f"Example {example_file} produced no output"


@pytest.mark.examples
def test_all_examples_present(examples_dir: Path) -> None:
    """Ensure all documented examples exist."""
    for example_file in EXAMPLE_FILES:
        example_path = examples_dir / example_file
        assert example_path.exists(), f"Missing example: {example_file}"


@pytest.mark.examples
@pytest.mark.parametrize("example_file", EXAMPLE_FILES)
def test_example_no_leftover_files(
    example_file: str,
    examples_dir: Path,
    tmp_path: Path,
) -> None:
    """Test that examples clean up after themselves."""
    example_path = examples_dir / example_file

    if not example_path.exists():
        pytest.skip(f"Example file not found: {example_path}")

    initial_files = set(tmp_path.rglob("*"))
    result = subprocess.run(
        [sys.executable, str(example_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )

    if result.returncode != 0:
        pytest.skip(f"Example failed to run: {result.stderr}")

    final_files = set(tmp_path.rglob("*"))
    leftover_files = final_files - initial_files
    leftover_files = {
        f
        for f in leftover_files
        if not any(
            part.startswith("__pycache__") or part.endswith(".pyc") for part in f.parts
        )
    }

    assert len(leftover_files) == 0, (
        f"Example {example_file} left files behind:\n"
        + "\n".join(f"  - {f}" for f in leftover_files)
    )


@pytest.mark.examples
@pytest.mark.parametrize(
    "example_file,expected_output",
    [
        ("config_file_migration.py", "SCENARIO"),
        ("message_queue_consumer.py", "Processing message"),
        ("etl_data_import.py", "IMPORT"),
        ("ml_inference_pipeline.py", "INFERENCE"),
        ("advanced_features.py", "MIGRATION"),
    ],
)
def test_example_expected_output(
    example_file: str,
    expected_output: str,
    examples_dir: Path,
    tmp_path: Path,
) -> None:
    """Test that examples produce expected output."""
    example_path = examples_dir / example_file

    if not example_path.exists():
        pytest.skip(f"Example file not found: {example_path}")

    result = subprocess.run(
        [sys.executable, str(example_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )

    assert result.returncode == 0, f"Example {example_file} failed to run"
    assert expected_output in result.stdout, (
        f"Example {example_file} did not produce expected output '{expected_output}'\n"
        f"Actual output:\n{result.stdout[:500]}"
    )


@pytest.mark.examples
def test_examples_directory_exists(source_root: Path) -> None:
    """Ensure examples directory exists."""
    examples_dir = source_root / "examples"
    assert examples_dir.exists(), "examples/ directory not found"
    assert examples_dir.is_dir(), "examples/ is not a directory"


@pytest.mark.examples
def test_all_python_files_in_examples_are_tested(examples_dir: Path) -> None:
    """Ensure we're not missing any examples."""
    if not examples_dir.exists():
        pytest.skip("Examples directory not found")

    actual_examples = {f.name for f in examples_dir.glob("*.py")}
    tested_examples = set(EXAMPLE_FILES)

    missing = actual_examples - tested_examples

    assert len(missing) == 0, f"Found {len(missing)} untested examples:\n" + "\n".join(
        f"  - {f}" for f in sorted(missing)
    )
