#!/usr/bin/env python3
"""Live human-readable view of a Claude Code stream-json log.

Usage:
    python3 scripts/watch_run.py <path>            # tail a file (default)
    python3 scripts/watch_run.py -                 # read from stdin
    python3 scripts/watch_run.py <path> -v         # include hooks + rate-limit + unknowns
    python3 scripts/watch_run.py <path> --no-color # plain text (force off)

Designed for watching a Claude Code `--output-format stream-json` log while
the agent is running. Renders one block per assistant message: agent text,
tool calls (name + truncated args), tool results (truncated). Drops session
hooks and rate-limit pings by default -- pass -v to keep them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

CLR: dict[str, str] = {}


def _setup_colors(enabled: bool) -> None:
    keys = ("reset", "bold", "gray", "green", "cyan", "magenta", "yellow", "red")
    if not enabled:
        for k in keys:
            CLR[k] = ""
        return
    CLR.update(
        reset="\x1b[0m", bold="\x1b[1m", gray="\x1b[90m",
        green="\x1b[32m", cyan="\x1b[36m", magenta="\x1b[35m",
        yellow="\x1b[33m", red="\x1b[31m",
    )


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"...({len(s) - n} more chars)"


def _flatten_tool_result_content(content: Any) -> str:
    """Tool result content can be a string or a list of blocks. Normalize to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for sub in content:
            if isinstance(sub, dict):
                t = sub.get("type")
                if t == "text":
                    out.append(sub.get("text", ""))
                elif t == "tool_reference":
                    out.append(f"<ref:{sub.get('tool_name')}>")
                else:
                    out.append(json.dumps(sub, ensure_ascii=False))
            else:
                out.append(str(sub))
        return "\n".join(out)
    return str(content)


def render(event: dict, verbose: bool) -> None:
    t = event.get("type")

    if t == "system":
        st = event.get("subtype")
        if st == "init":
            sid = (event.get("session_id") or "?")[:8]
            model = event.get("model", "?")
            cwd = event.get("cwd", "?")
            print(f"{CLR['gray']}--- session {sid} | model={model} | cwd={cwd}{CLR['reset']}")
        elif verbose and st in ("hook_started", "hook_response"):
            print(f"{CLR['gray']}  [hook {st}] {event.get('hook_name', '?')} exit={event.get('exit_code', '')}{CLR['reset']}")

    elif t == "assistant":
        msg = event.get("message", {})
        for c in msg.get("content", []):
            ct = c.get("type")
            if ct == "text":
                text = (c.get("text") or "").strip()
                if text:
                    print(f"{CLR['bold']}{CLR['cyan']}[assistant]{CLR['reset']} {text}")
            elif ct == "tool_use":
                name = c.get("name", "?")
                inp_repr = json.dumps(c.get("input", {}), ensure_ascii=False)
                print(f"{CLR['magenta']}  [tool] {name}{CLR['reset']} {CLR['gray']}{_truncate(inp_repr, 200)}{CLR['reset']}")
        u = msg.get("usage") or {}
        if u.get("output_tokens") or u.get("input_tokens"):
            print(
                f"{CLR['gray']}    [tokens] in={u.get('input_tokens', 0)} "
                f"cache_read={u.get('cache_read_input_tokens', 0)} "
                f"out={u.get('output_tokens', 0)}{CLR['reset']}"
            )

    elif t == "user":
        msg = event.get("message", {})
        for c in msg.get("content", []):
            if c.get("type") != "tool_result":
                continue
            body = _flatten_tool_result_content(c.get("content", "")).strip()
            is_error = bool(c.get("is_error"))
            color = CLR["red"] if is_error else CLR["green"]
            tag = "tool_error" if is_error else "tool_result"
            print(f"{color}  [{tag}]{CLR['reset']} {_truncate(body, 500)}")

    elif t == "result":
        ok = event.get("subtype") == "success"
        cost = event.get("total_cost_usd", 0) or 0
        dur_s = (event.get("duration_ms") or 0) / 1000
        color = CLR["yellow"] if ok else CLR["red"]
        print(
            f"{CLR['bold']}{color}=== run finished | ok={ok} | "
            f"duration={dur_s:.0f}s | cost=${cost:.2f}{CLR['reset']}"
        )

    elif t == "rate_limit_event":
        if verbose:
            info = event.get("rate_limit_info") or {}
            print(f"{CLR['gray']}  [rate_limit] status={info.get('status')} overage={info.get('overageStatus')}{CLR['reset']}")

    else:
        if verbose:
            print(f"{CLR['gray']}  [{t}] {_truncate(json.dumps(event, ensure_ascii=False), 200)}{CLR['reset']}")


def stream_file(path: Path, verbose: bool) -> None:
    """tail -F equivalent: keep reading; handle truncation/rotation by reopening on inode change."""
    while not path.exists():
        time.sleep(0.5)
    fh = path.open("r")
    inode = path.stat().st_ino
    while True:
        line = fh.readline()
        if line:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue  # log may contain non-JSON noise (echoes, progress prints); skip
            render(ev, verbose)
            sys.stdout.flush()
        else:
            time.sleep(0.5)
            try:
                cur_inode = path.stat().st_ino
            except FileNotFoundError:
                time.sleep(1)
                continue
            if cur_inode != inode:
                fh.close()
                fh = path.open("r")
                inode = cur_inode


def stream_stdin(verbose: bool) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        render(ev, verbose)
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live pretty-printer for Claude Code stream-json logs.")
    ap.add_argument("input", help="path to .jsonl log, or '-' for stdin")
    ap.add_argument("-v", "--verbose", action="store_true", help="include hooks + rate-limit + unknown events")
    ap.add_argument("--no-color", action="store_true", help="force-disable ANSI color")
    ap.add_argument("--color", action="store_true", help="force-enable ANSI color (override tty check)")
    args = ap.parse_args()

    if args.no_color:
        enabled = False
    elif args.color:
        enabled = True
    else:
        enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    _setup_colors(enabled)

    if args.input == "-":
        stream_stdin(args.verbose)
    else:
        stream_file(Path(args.input), args.verbose)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
