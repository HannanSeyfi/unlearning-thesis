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
import time
from urllib.parse import quote
from base64 import b64encode
from pathlib import Path
from typing import Iterable, Sequence

import requests


DEFAULT_REPOSITORY = "HannanSeyfi/unlearning-thesis"
DEFAULT_BRANCH = "main"
DEFAULT_REPO_DIR = Path("/content/unlearning-thesis")
DEFAULT_GIT_USER_NAME = "Colab Thesis Runner"
DEFAULT_GIT_USER_EMAIL = "colab-thesis-runner@example.com"
GITHUB_MAX_FILE_MB = 95
GIT_PUSH_MAX_ATTEMPTS = 4
GIT_PUSH_RETRY_BASE_SECONDS = 2
DEFAULT_RESUME_RELEASE_TAG = "week5-resume-state"
DEFAULT_RESUME_RELEASE_NAME = "Week 5 Resume State"


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


def github_api_request(
    method: str,
    url: str,
    *,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    allowed_statuses: Sequence[int] = (),
    **kwargs,
) -> requests.Response:
    """Call the GitHub REST API with notebook credentials."""
    token = token or read_github_token(required=True)
    request_headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if headers:
        request_headers.update(headers)
    response = requests.request(method, url, headers=request_headers, **kwargs)
    if response.status_code >= 400 and response.status_code not in allowed_statuses:
        raise RuntimeError(
            f"GitHub API request failed: {method} {url}\n"
            f"Status: {response.status_code}\n"
            f"{response.text[:2000]}"
        )
    return response


def get_release_by_tag(
    repository: str = DEFAULT_REPOSITORY,
    tag: str = DEFAULT_RESUME_RELEASE_TAG,
    *,
    token: str | None = None,
    required: bool = True,
) -> dict | None:
    """Return a release by tag, optionally allowing it to be missing."""
    url = f"https://api.github.com/repos/{repository}/releases/tags/{quote(tag)}"
    response = github_api_request(
        "GET",
        url,
        token=token,
        allowed_statuses=(() if required else (404,)),
    )
    if response.status_code == 404:
        return None
    return response.json()


def get_or_create_release(
    repository: str = DEFAULT_REPOSITORY,
    tag: str = DEFAULT_RESUME_RELEASE_TAG,
    *,
    name: str = DEFAULT_RESUME_RELEASE_NAME,
    target_branch: str = DEFAULT_BRANCH,
    token: str | None = None,
) -> dict:
    """Return the resume release, creating it if needed."""
    token = token or read_github_token(required=True)
    release = get_release_by_tag(repository, tag, token=token, required=False)
    if release:
        return release

    url = f"https://api.github.com/repos/{repository}/releases"
    response = github_api_request(
        "POST",
        url,
        token=token,
        json={
            "tag_name": tag,
            "target_commitish": target_branch,
            "name": name,
            "body": "Machine-readable Colab resume assets. Not a thesis result release.",
            "draft": False,
            "prerelease": True,
        },
    )
    return response.json()


def list_release_assets(release: dict, *, token: str | None = None) -> list[dict]:
    """List all assets for a release."""
    assets: list[dict] = []
    url = release["assets_url"] + "?per_page=100"
    while url:
        response = github_api_request("GET", url, token=token)
        assets.extend(response.json())
        url = response.links.get("next", {}).get("url")
    return assets


def upload_release_asset(
    path: str | Path,
    asset_name: str,
    *,
    repository: str = DEFAULT_REPOSITORY,
    release_tag: str = DEFAULT_RESUME_RELEASE_TAG,
    token: str | None = None,
) -> dict:
    """Upload or replace a binary release asset."""
    token = token or read_github_token(required=True)
    path = Path(path)
    release = get_or_create_release(repository, release_tag, token=token)

    for asset in list_release_assets(release, token=token):
        if asset["name"] == asset_name:
            github_api_request("DELETE", asset["url"], token=token)
            break

    upload_base = release["upload_url"].split("{", 1)[0]
    upload_url = f"{upload_base}?name={quote(asset_name)}"
    with path.open("rb") as handle:
        response = github_api_request(
            "POST",
            upload_url,
            token=token,
            headers={
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/octet-stream",
            },
            data=handle,
        )
    print("Uploaded GitHub release asset:", asset_name)
    return response.json()


def download_release_asset(
    asset_name: str,
    output_path: str | Path,
    *,
    repository: str = DEFAULT_REPOSITORY,
    release_tag: str = DEFAULT_RESUME_RELEASE_TAG,
    token: str | None = None,
    required: bool = True,
) -> bool:
    """Download a binary release asset by name."""
    token = token or read_github_token(required=True)
    release = get_release_by_tag(repository, release_tag, token=token, required=required)
    if not release:
        return False

    match = None
    for asset in list_release_assets(release, token=token):
        if asset["name"] == asset_name:
            match = asset
            break

    if not match:
        if required:
            raise FileNotFoundError(f"Release asset not found: {asset_name}")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = github_api_request(
        "GET",
        match["url"],
        token=token,
        headers={"Accept": "application/octet-stream"},
        stream=True,
    )
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    print("Downloaded GitHub release asset:", asset_name)
    return True


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

    restore_training_state_release_assets(repo_dir=repo_dir)
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


def _training_state_asset_name(path: Path, repo_dir: Path) -> str:
    rel_path = _relative_to_repo(path, repo_dir)
    safe_name = rel_path.replace("/", "__").replace(" ", "_")
    return f"resume__{safe_name}"


def _iter_training_state_paths(paths: list[Path]) -> list[Path]:
    training_state_paths: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "training_state.pt":
            training_state_paths.append(path)
        elif path.is_dir():
            training_state_paths.extend(path.rglob("training_state.pt"))
    return training_state_paths


def sync_training_state_release_assets(
    paths: list[Path],
    *,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    release_tag: str = DEFAULT_RESUME_RELEASE_TAG,
) -> None:
    """Upload ignored optimizer states and annotate latest.json for resume."""
    repo_dir = Path(repo_dir)
    for state_path in _iter_training_state_paths(paths):
        if not state_path.exists():
            continue

        latest_path = state_path.parent.parent / "latest.json"
        if not latest_path.exists():
            continue

        import json

        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        latest_state_path = latest.get("training_state_path")
        if latest_state_path and Path(latest_state_path).resolve() != state_path.resolve():
            continue
        if latest.get("training_state_release_asset"):
            continue

        asset_name = _training_state_asset_name(state_path, repo_dir)
        upload_release_asset(state_path, asset_name, release_tag=release_tag)

        latest["training_state_release_asset"] = asset_name
        latest["training_state_release_tag"] = release_tag
        latest_path.write_text(
            json.dumps(latest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def restore_training_state_release_assets(
    *,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    release_tag: str = DEFAULT_RESUME_RELEASE_TAG,
) -> None:
    """Restore missing optimizer states from resume release assets."""
    import json

    repo_dir = Path(repo_dir)
    for latest_path in repo_dir.glob(
        "Week 5/results/*/resume_state/epoch_checkpoints/*/latest.json"
    ):
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        asset_name = latest.get("training_state_release_asset")
        state_path_value = latest.get("training_state_path")
        if not asset_name or not state_path_value:
            continue

        state_path = Path(state_path_value)
        if state_path.exists():
            continue

        download_release_asset(
            asset_name,
            state_path,
            release_tag=latest.get("training_state_release_tag", release_tag),
            required=False,
        )


def untrack_ignored_files(*, repo_dir: str | Path = DEFAULT_REPO_DIR) -> list[str]:
    """Stage removals for files that are tracked but now ignored."""
    repo_dir = Path(repo_dir)
    ignored = run_git(
        ["ls-files", "-i", "--exclude-standard", "-c", "-z"],
        cwd=repo_dir,
        check=False,
    )
    files = [path for path in ignored.stdout.split("\0") if path]
    if files:
        run_git(["rm", "--cached", "--ignore-unmatch", "-r", "--", *files], cwd=repo_dir)
        print("Untracked ignored files:", len(files))
    return files


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
    sync_training_state_release_assets(existing_paths, repo_dir=repo_dir)
    untrack_ignored_files(repo_dir=repo_dir)

    rel_paths = [_relative_to_repo(path, repo_dir) for path in existing_paths]
    run_git(["add", "--", *rel_paths], cwd=repo_dir)

    staged = run_git(["diff", "--cached", "--quiet"], cwd=repo_dir, check=False)
    has_staged_changes = staged.returncode != 0
    if not has_staged_changes and not allow_empty:
        ahead = run_git(
            ["rev-list", "--count", f"origin/{branch}..HEAD"],
            cwd=repo_dir,
            check=False,
        )
        ahead_count = int(ahead.stdout.strip() or "0") if ahead.returncode == 0 else 0
        if ahead_count == 0:
            print("No GitHub output changes to commit.")
            return False
        print(f"No new file changes; pushing {ahead_count} queued commit(s).")
    else:
        commit_args = ["commit", "-m", message]
        if allow_empty:
            commit_args.insert(1, "--allow-empty")
        run_git(commit_args, cwd=repo_dir)

    last_push_output = ""
    for attempt in range(1, GIT_PUSH_MAX_ATTEMPTS + 1):
        push = run_git(["push", "origin", branch], cwd=repo_dir, token=token, check=False)
        if push.returncode == 0:
            print("Pushed outputs to GitHub.")
            print(push.stdout.strip())
            return True

        last_push_output = push.stdout.strip()
        if attempt == GIT_PUSH_MAX_ATTEMPTS:
            break

        if "non-fast-forward" in last_push_output or "fetch first" in last_push_output:
            rebase = run_git(
                ["pull", "--rebase", "--autostash", "origin", branch],
                cwd=repo_dir,
                token=token,
                check=False,
            )
            if rebase.returncode != 0:
                print("Git pull before push retry also failed:")
                print(rebase.stdout.strip())

        delay_seconds = GIT_PUSH_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
        print(
            f"Git push attempt {attempt}/{GIT_PUSH_MAX_ATTEMPTS} failed; "
            f"retrying in {delay_seconds}s."
        )
        print(last_push_output)
        time.sleep(delay_seconds)

    raise RuntimeError(
        f"Command failed after {GIT_PUSH_MAX_ATTEMPTS} attempts: git push origin {branch}\n"
        f"{last_push_output}"
    )
