#!/usr/bin/env python3
"""Colab-to-GitHub smoke test.

This script creates or updates test-runs/HelloWorld.txt in the GitHub repo.
It is intentionally tiny so we can confirm that Colab can write results back
to GitHub before changing the week-by-week notebooks.

In Google Colab, add a secret named GITHUB_TOKEN with repository Contents
read/write access, then run this file.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_REPOSITORY = "HannanSeyfi/unlearning-thesis"
DEFAULT_BRANCH = "main"
DEFAULT_OUTPUT_PATH = "test-runs/HelloWorld.txt"


def read_github_token() -> str | None:
    """Read the token from env vars first, then from Colab Secrets."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token.strip()

    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None

    try:
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        return None

    return token.strip() if token else None


def build_hello_world_text(repository: str, branch: str, output_path: str) -> str:
    utc_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return "\n".join(
        [
            "Hello, World!",
            "",
            "This file was written by test-runs/github_hello_world.py.",
            f"UTC time: {utc_now}",
            f"Repository: {repository}",
            f"Branch: {branch}",
            f"Output path: {output_path}",
            f"Python: {platform.python_version()}",
            f"Platform: {platform.platform()}",
            "",
        ]
    )


def github_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "unlearning-thesis-colab-smoke-test",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def api_error_message(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    message = payload.get("message")
    return str(message if message else payload)


def decode_json_response(raw_body: bytes) -> Any:
    if not raw_body:
        return {}

    text = raw_body.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}


def github_api_json(
    *,
    method: str,
    url: str,
    token: str,
    params: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    if params:
        url = f"{url}?{urlencode(params)}"

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = github_headers(token)
    if payload is not None:
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(request, timeout=30) as response:
            return response.status, decode_json_response(response.read())
    except HTTPError as exc:
        return exc.code, decode_json_response(exc.read())
    except URLError as exc:
        raise RuntimeError(f"Could not connect to GitHub: {exc.reason}") from exc


def get_existing_file_sha(
    *,
    token: str,
    repository: str,
    branch: str,
    output_path: str,
) -> str | None:
    encoded_path = quote(output_path, safe="/")
    url = f"https://api.github.com/repos/{repository}/contents/{encoded_path}"
    status_code, payload = github_api_json(
        method="GET",
        url=url,
        token=token,
        params={"ref": branch},
    )

    if status_code == 404:
        return None

    if status_code >= 400:
        raise RuntimeError(
            f"Could not inspect {output_path}: "
            f"GitHub API {status_code}: {api_error_message(payload)}"
        )

    sha = payload.get("sha")
    return str(sha) if sha else None


def write_file_to_github(
    *,
    token: str,
    repository: str,
    branch: str,
    output_path: str,
    content: str,
    commit_message: str,
) -> dict[str, Any]:
    existing_sha = get_existing_file_sha(
        token=token,
        repository=repository,
        branch=branch,
        output_path=output_path,
    )

    encoded_path = quote(output_path, safe="/")
    url = f"https://api.github.com/repos/{repository}/contents/{encoded_path}"
    payload: dict[str, Any] = {
        "branch": branch,
        "message": commit_message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }

    if existing_sha:
        payload["sha"] = existing_sha

    status_code, response_payload = github_api_json(
        method="PUT",
        url=url,
        token=token,
        payload=payload,
    )

    if status_code not in {200, 201}:
        raise RuntimeError(
            f"Could not write {output_path}: "
            f"GitHub API {status_code}: {api_error_message(response_payload)}"
        )

    return response_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update HelloWorld.txt in this GitHub repository."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY),
        help=f"GitHub repository in owner/name form. Default: {DEFAULT_REPOSITORY}",
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("GITHUB_BRANCH", DEFAULT_BRANCH),
        help=f"Branch to write to. Default: {DEFAULT_BRANCH}",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("GITHUB_OUTPUT_PATH", DEFAULT_OUTPUT_PATH),
        help=f"Repository path to create or update. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--message",
        default="Colab smoke test: write HelloWorld.txt",
        help="Commit message for the GitHub file update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the file content without calling the GitHub API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    content = build_hello_world_text(args.repo, args.branch, args.path)

    if args.dry_run:
        print("Dry run only. No GitHub API call was made.")
        print(f"Target: https://github.com/{args.repo}/blob/{args.branch}/{args.path}")
        print()
        print(content)
        return 0

    token = read_github_token()
    if not token:
        print(
            "Missing GITHUB_TOKEN.\n"
            "In Colab, open Secrets, add GITHUB_TOKEN, and use a GitHub token "
            "with Contents read/write access for this repository.",
            file=sys.stderr,
        )
        return 2

    result = write_file_to_github(
        token=token,
        repository=args.repo,
        branch=args.branch,
        output_path=args.path,
        content=content,
        commit_message=args.message,
    )

    commit = result.get("commit", {})
    content_info = result.get("content", {})
    commit_sha = commit.get("sha", "unknown")
    html_url = content_info.get(
        "html_url", f"https://github.com/{args.repo}/blob/{args.branch}/{args.path}"
    )

    print("Success: HelloWorld.txt was written to GitHub.")
    print(f"File: {html_url}")
    print(f"Commit: {commit_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
