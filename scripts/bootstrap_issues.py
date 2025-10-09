"""Create GitHub issues from ops/issues.yml and optionally add to a Project (v2).

Requirements:
- Python 3.10+
- PyYAML (already in project deps)
- GitHub CLI (`gh`) authenticated (GH_TOKEN or `gh auth login`)

Usage:
  python scripts/bootstrap_issues.py --repo OWNER/REPO [--owner ORG_OR_USER --project-number N]

Notes:
- If --owner and --project-number are provided, the script will try to add created issues to the
  specified GitHub Project (v2). A Status field named "Status" is assumed or will be created with
  options Backlog/In Progress/Done, and items are set to the issue's `status` value from YAML.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str], check: bool = True, capture: bool = True) -> str:
    res = subprocess.run(cmd, check=check, capture_output=capture, text=True)
    return (res.stdout or "").strip()


def ensure_gh() -> None:
    if shutil.which("gh") is None:
        print("GitHub CLI 'gh' is required. Install from https://cli.github.com/ and auth with 'gh auth login'.", file=sys.stderr)
        sys.exit(2)


def load_issues_spec() -> List[Dict[str, Any]]:
    spec_path = ROOT / "ops" / "issues.yml"
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    return list(data.get("issues", []))


def create_label(repo: str, name: str) -> None:
    try:
        run(["gh", "label", "create", name, "--repo", repo])
    except subprocess.CalledProcessError:
        # likely exists
        pass


def get_or_create_status_field(owner: str, project_number: int) -> Dict[str, Any]:
    # List fields
    out = run(["gh", "project", "field-list", f"{owner}/{project_number}", "--format", "json"])
    fields = json.loads(out) if out else []
    for f in fields:
        if f.get("name") == "Status":
            return f
    # Create Status field with options
    run(["gh", "project", "field-create", f"{owner}/{project_number}", "--name", "Status", "--data-type", "SINGLE_SELECT"])  # noqa: E501
    out2 = run(["gh", "project", "field-list", f"{owner}/{project_number}", "--format", "json"])
    fields2 = json.loads(out2) if out2 else []
    for f in fields2:
        if f.get("name") == "Status":
            # Ensure options exist
            for opt in ("Backlog", "In Progress", "Done"):
                try:
                    run(["gh", "project", "field-update", f"{owner}/{project_number}", "--name", "Status", "--single-select-options", opt])  # noqa: E501
                except subprocess.CalledProcessError:
                    pass
            return f
    raise RuntimeError("Failed to create/find Status field")


def option_id_for(field: Dict[str, Any], option_name: str) -> str | None:
    for opt in field.get("options", []):
        if opt.get("name") == option_name:
            return opt.get("id")
    return None


def add_issue_to_project(owner: str, project_number: int, issue_url: str, status: str) -> None:
    # Add item
    out = run(["gh", "project", "item-add", f"{owner}/{project_number}", "--url", issue_url, "--format", "json"])  # noqa: E501
    item = json.loads(out) if out else {}
    item_id = item.get("id")
    if not item_id:
        # Fallback: try to find by URL
        items_json = run(["gh", "project", "item-list", f"{owner}/{project_number}", "--format", "json"])
        items = json.loads(items_json) if items_json else []
        for it in items:
            content = it.get("content") or {}
            if content.get("url") == issue_url:
                item_id = it.get("id")
                break
    if not item_id:
        print(f"Warning: could not resolve project item for {issue_url}")
        return
    field = get_or_create_status_field(owner, project_number)
    opt_id = option_id_for(field, status) or option_id_for(field, "Backlog")
    if opt_id:
        run([
            "gh", "project", "item-edit", f"{owner}/{project_number}",
            "--id", item_id,
            "--field-id", field["id"],
            "--single-select-option-id", opt_id,
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="OWNER/REPO, e.g., org/repo")
    parser.add_argument("--owner", help="Project owner (org or user) for Projects v2")
    parser.add_argument("--project-number", type=int, help="Project number for Projects v2")
    args = parser.parse_args()

    ensure_gh()
    issues = load_issues_spec()

    # Ensure labels
    labels = set(l for it in issues for l in it.get("labels", []))
    for lb in sorted(labels):
        create_label(args.repo, lb)

    # Create issues
    created: List[Dict[str, str]] = []
    for it in issues:
        title = it["title"]
        body = it.get("body", "")
        lbls = [l for l in it.get("labels", [])]
        cmd = ["gh", "issue", "create", "--repo", args.repo, "--title", title]
        if body:
            cmd += ["--body", body]
        for lb in lbls:
            cmd += ["--label", lb]
        out = run(cmd)
        url = out.strip().splitlines()[-1]
        created.append({"title": title, "url": url, "status": it.get("status", "Backlog")})

    # Optionally add to project
    if args.owner and args.project_number:
        for it in created:
            add_issue_to_project(args.owner, args.project_number, it["url"], it["status"]) 

    print(f"Created {len(created)} issues.")


if __name__ == "__main__":
    main()

