"""GitHub-backed Colab workspace helpers for the thesis notebooks.

The notebooks use these helpers instead of Google Drive:

1. Clone or update the GitHub repo inside Colab.
2. Write outputs under the cloned repo path.
3. Commit and push the changed output folders back to GitHub.

Keep the GitHub token in Colab Secrets as GITHUB_TOKEN. Do not paste it into a
notebook cell.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from base64 import b64encode
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_REPOSITORY = "HannanSeyfi/unlearning-thesis"
DEFAULT_BRANCH = "main"
DEFAULT_REPO_DIR = Path("/content/unlearning-thesis")
DEFAULT_GIT_USER_NAME = "Colab Thesis Runner"
DEFAULT_GIT_USER_EMAIL = "colab-thesis-runner@example.com"
GITHUB_MAX_FILE_MB = 95


def read_github_token(required: bool = True) -> str | None:
    """Read GITHUB_TOKEN from env vars or Colab Secrets."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            from google.colab import userdata  # type: ignore

            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            token = None

    if token:
        token = token.strip()
        os.environ["GITHUB_TOKEN"] = token
        return token

    if required:
        raise RuntimeError(
            "Missing GITHUB_TOKEN. In Colab, open Secrets, add GITHUB_TOKEN, "
            "and give the notebook access to it."
        )
    return None


def _auth_git_args(token: str | None) -> list[str]:
    if not token:
        return []
    credential = b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    return [
        "-c",
        f"http.https://github.com/.extraheader=AUTHORIZATION: basic {credential}",
    ]


def run_git(
    args: Sequence[str],
    *,
    cwd: Path | str | None = None,
    token: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run git with optional GitHub token auth and useful error output."""
    command = ["git", *_auth_git_args(token), *args]
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
        safe_command = "git " + " ".join(args)
        raise RuntimeError(
            f"Command failed: {safe_command}\n"
            f"Exit code: {result.returncode}\n"
            f"{result.stdout}"
        )
    return result


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _is_under_content(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path("/content").resolve())
        return True
    except ValueError:
        return False


def setup_colab_repo(
    repository: str = DEFAULT_REPOSITORY,
    branch: str = DEFAULT_BRANCH,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    *,
    replace_non_git_dir: bool = True,
) -> Path:
    """Clone or update the GitHub repo and return its local path."""
    token = read_github_token(required=True)
    repo_dir = Path(repo_dir)
    repo_url = f"https://github.com/{repository}.git"

    if repo_dir.exists() and not _is_git_repo(repo_dir):
        if replace_non_git_dir and _is_under_content(repo_dir):
            shutil.rmtree(repo_dir)
        else:
            raise RuntimeError(f"{repo_dir} exists but is not a git repository.")

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {repository} into {repo_dir} ...")
        run_git(
            ["clone", "--branch", branch, repo_url, str(repo_dir)],
            cwd=repo_dir.parent,
            token=token,
        )
    else:
        print(f"Using existing repo at {repo_dir}")

    run_git(["remote", "set-url", "origin", repo_url], cwd=repo_dir)
    run_git(["config", "user.name", DEFAULT_GIT_USER_NAME], cwd=repo_dir)
    run_git(["config", "user.email", DEFAULT_GIT_USER_EMAIL], cwd=repo_dir)
    run_git(["checkout", branch], cwd=repo_dir)

    status = run_git(["status", "--porcelain"], cwd=repo_dir, check=False)
    if not status.stdout.strip():
        run_git(["pull", "--rebase", "origin", branch], cwd=repo_dir, token=token)
    else:
        print("Local repo has changes; skipping pull before notebook writes.")

    print("GitHub-backed thesis folder:", repo_dir)
    return repo_dir


def resolve_week35_baseline_dir(thesis_dir: Path | str) -> Path:
    """Return the preferred Week 3.5 baseline run, falling back to the archive."""
    thesis_dir = Path(thesis_dir)
    week35_results = thesis_dir / "Week 3.5" / "results"
    expected = week35_results / "qwen05_high_accuracy_baseline"
    archived = week35_results / "reference_successful_run"
    if expected.exists():
        return expected
    if archived.exists():
        print("Using archived Week 3.5 reference run:", archived)
        return archived
    return expected


def _normalise_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _relative_to_repo(path: Path, repo_dir: Path) -> str:
    try:
        return path.resolve().relative_to(repo_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside repo: {path}") from exc


def _large_files(paths: list[Path], max_file_mb: int) -> list[tuple[Path, float]]:
    too_large: list[tuple[Path, float]] = []
    max_bytes = max_file_mb * 1024 * 1024
    for path in paths:
        if path.is_file() and path.stat().st_size > max_bytes:
            too_large.append((path, path.stat().st_size / (1024 * 1024)))
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.stat().st_size > max_bytes:
                    too_large.append((file_path, file_path.stat().st_size / (1024 * 1024)))
    return too_large


def commit_and_push(
    paths: str | Path | Iterable[str | Path],
    message: str,
    *,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    branch: str = DEFAULT_BRANCH,
    max_file_mb: int = GITHUB_MAX_FILE_MB,
    allow_empty: bool = False,
) -> bool:
    """Commit selected repo paths and push them to GitHub."""
    token = read_github_token(required=True)
    repo_dir = Path(repo_dir)
    selected_paths = _normalise_paths(paths)
    existing_paths = [path for path in selected_paths if path.exists()]
    missing_paths = [path for path in selected_paths if not path.exists()]

    for path in missing_paths:
        print("Skipping missing path:", path)

    if not existing_paths and not allow_empty:
        print("No existing output paths to commit.")
        return False

    too_large = _large_files(existing_paths, max_file_mb)
    if too_large:
        details = "\n".join(
            f"- {_relative_to_repo(path, repo_dir)} ({size_mb:.1f} MB)"
            for path, size_mb in too_large
        )
        raise RuntimeError(
            f"GitHub rejects files larger than about 100 MB. "
            f"These files exceed the {max_file_mb} MB safety limit:\n{details}"
        )

    run_git(["pull", "--rebase", "--autostash", "origin", branch], cwd=repo_dir, token=token)

    rel_paths = [_relative_to_repo(path, repo_dir) for path in existing_paths]
    run_git(["add", "--", *rel_paths], cwd=repo_dir)

    staged = run_git(["diff", "--cached", "--quiet"], cwd=repo_dir, check=False)
    if staged.returncode == 0 and not allow_empty:
        print("No GitHub output changes to commit.")
        return False

    commit_args = ["commit", "-m", message]
    if allow_empty:
        commit_args.insert(1, "--allow-empty")
    run_git(commit_args, cwd=repo_dir)

    push = run_git(["push", "origin", branch], cwd=repo_dir, token=token, check=False)
    if push.returncode == 0:
        print("Pushed outputs to GitHub.")
        print(push.stdout.strip())
        return True

    print("Initial push failed; rebasing once and retrying.")
    print(push.stdout.strip())
    run_git(["pull", "--rebase", "--autostash", "origin", branch], cwd=repo_dir, token=token)
    run_git(["push", "origin", branch], cwd=repo_dir, token=token)
    print("Pushed outputs to GitHub after rebase.")
    return True
