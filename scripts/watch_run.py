#!/usr/bin/env python3
"""Live human-readable view of a Claude Code stream-json log.

Usage:
    python3 scripts/watch_run.py <path>          # tail a file (default)
    python3 scripts/watch_run.py -               # read from stdin
    python3 scripts/watch_run.py <path> -v       # include hooks + rate-limit + unknowns
    python3 scripts/watch_run.py <path> -t       # text-only (hide tool calls + results)
    python3 scripts/watch_run.py <path> --no-color
    python3 scripts/watch_run.py <path> --max-result-lines 4

Renders one block per assistant TURN (a single message id). Each turn:
  ─── 14:25:33  turn 12  cum_in=137  cache=98k  cum_out=872 ───────────────
    <assistant prose, wrapped to terminal width>
    -> Bash (commit data prep): git add train.py && git commit -m "..."
    <- [evo/run_0000/exp_0000 165e830] add: ...
       2 files changed
    -> Write /path/to/file.py: 5716 chars
    <- File created successfully
Tool-specific renderers strip the JSON noise for the common tools
(Bash, Write, Edit, Read, Skill, Glob, Grep, TodoWrite, ToolSearch, Task).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

CLR: dict[str, str] = {}


def _setup_colors(enabled: bool) -> None:
    keys = ("reset", "bold", "dim", "gray", "red", "green", "yellow",
            "blue", "magenta", "cyan", "white")
    if not enabled:
        for k in keys:
            CLR[k] = ""
        return
    CLR.update(
        reset="\x1b[0m", bold="\x1b[1m", dim="\x1b[2m",
        gray="\x1b[90m", red="\x1b[31m", green="\x1b[32m",
        yellow="\x1b[33m", blue="\x1b[34m", magenta="\x1b[35m",
        cyan="\x1b[36m", white="\x1b[37m",
    )


class State:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.turn = 0
        self.term_width = args.width or shutil.get_terminal_size((100, 30)).columns
        self.last_assistant_msg_id: str | None = None
        self.cum_in = 0
        self.cum_out = 0
        self.cum_cache_read = 0


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _hr(state: State, label: str = "") -> None:
    w = max(20, state.term_width - 1)
    if label:
        dash_count = max(3, w - len(label) - 5)
        line = f"--- {label} " + "-" * dash_count
    else:
        line = "-" * w
    print(f"{CLR['dim']}{line}{CLR['reset']}")


def _wrap(text: str, width: int, initial_indent: str = "", subsequent_indent: str = "") -> str:
    out = []
    for line in text.split("\n"):
        if not line.strip():
            out.append(line)
            continue
        wrapped = textwrap.fill(
            line, width=max(40, width),
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        out.append(wrapped or line)
    return "\n".join(out)


def _truncate_lines(s: str, max_lines: int, max_chars_per_line: int) -> Tuple[str, int]:
    lines = s.split("\n")
    truncated_count = 0
    if len(lines) > max_lines:
        truncated_count = len(lines) - max_lines
        lines = lines[:max_lines]
    out = []
    for ln in lines:
        if len(ln) > max_chars_per_line:
            out.append(ln[: max_chars_per_line - 1] + "…")
        else:
            out.append(ln)
    return "\n".join(out), truncated_count


# Tool-specific input formatters.  Return (label, detail).
# `label` goes on the same line as the arrow; `detail` is shown after a colon
# (or wrapped to the next line if it's multi-line).
def _fmt_bash(inp: dict) -> Tuple[str, str]:
    cmd = inp.get("command", "")
    desc = inp.get("description", "")
    label = f"Bash ({desc})" if desc else "Bash"
    return label, cmd


def _fmt_write(inp: dict) -> Tuple[str, str]:
    path = inp.get("file_path", "?")
    content = inp.get("content", "")
    return f"Write {path}", f"{len(content)} chars"


def _fmt_edit(inp: dict) -> Tuple[str, str]:
    path = inp.get("file_path", "?")
    old = inp.get("old_string", "")
    new = inp.get("new_string", "")
    return f"Edit {path}", f"{len(old)} -> {len(new)} chars"


def _fmt_read(inp: dict) -> Tuple[str, str]:
    path = inp.get("file_path", "?")
    offset = inp.get("offset")
    limit = inp.get("limit")
    rng = ""
    if offset is not None or limit is not None:
        start = offset or 0
        end = start + (limit or 0) if limit else "?"
        rng = f" lines {start}-{end}"
    return f"Read {path}{rng}", ""


def _fmt_skill(inp: dict) -> Tuple[str, str]:
    return f"Skill {inp.get('skill', '?')}", inp.get("args", "") or ""


def _fmt_glob(inp: dict) -> Tuple[str, str]:
    return f"Glob {inp.get('pattern', '?')}", inp.get("path", "") or ""


def _fmt_grep(inp: dict) -> Tuple[str, str]:
    pat = inp.get("pattern", "?")
    path = inp.get("path", "")
    extra = []
    if inp.get("type"):
        extra.append(f"type={inp['type']}")
    if inp.get("-i"):
        extra.append("-i")
    detail = " ".join([path] + extra) if path or extra else ""
    return f"Grep '{pat}'", detail


def _fmt_todowrite(inp: dict) -> Tuple[str, str]:
    todos = inp.get("todos", []) or []
    active = sum(1 for t in todos if t.get("status") == "in_progress")
    done = sum(1 for t in todos if t.get("status") == "completed")
    pending = sum(1 for t in todos if t.get("status") == "pending")
    return "TodoWrite", f"{len(todos)} todos ({done} done, {active} active, {pending} pending)"


def _fmt_toolsearch(inp: dict) -> Tuple[str, str]:
    return "ToolSearch", inp.get("query", "") or ""


def _fmt_task(inp: dict) -> Tuple[str, str]:
    desc = inp.get("description", "")
    sub = inp.get("subagent_type", "")
    prompt = inp.get("prompt", "")
    label = f"Task ({sub})" if sub else "Task"
    detail = desc or (prompt[:200] + "..." if len(prompt) > 200 else prompt)
    return label, detail


def _fmt_generic(name: str, inp: dict) -> Tuple[str, str]:
    s = json.dumps(inp, ensure_ascii=False)
    if len(s) > 200:
        s = s[:200] + f"... ({len(s) - 200} more)"
    return name, s


TOOL_FORMATTERS = {
    "Bash": _fmt_bash,
    "Write": _fmt_write,
    "Edit": _fmt_edit,
    "Read": _fmt_read,
    "Skill": _fmt_skill,
    "Glob": _fmt_glob,
    "Grep": _fmt_grep,
    "TodoWrite": _fmt_todowrite,
    "ToolSearch": _fmt_toolsearch,
    "Task": _fmt_task,
    "Agent": _fmt_task,
}


def _fmt_tool(name: str, inp: dict) -> Tuple[str, str]:
    fn = TOOL_FORMATTERS.get(name)
    if fn:
        return fn(inp)
    return _fmt_generic(name, inp)


def _flatten_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for sub in content:
            if isinstance(sub, dict):
                t = sub.get("type")
                if t == "text":
                    parts.append(sub.get("text", ""))
                elif t == "tool_reference":
                    parts.append(f"<ref:{sub.get('tool_name')}>")
                else:
                    parts.append(json.dumps(sub, ensure_ascii=False))
            else:
                parts.append(str(sub))
        return "\n".join(parts)
    return str(content)


def render(event: dict, state: State) -> None:
    t = event.get("type")

    if t == "system":
        st = event.get("subtype")
        if st == "init":
            sid = (event.get("session_id") or "?")[:8]
            model = event.get("model", "?")
            cwd = event.get("cwd", "?")
            _hr(state, f"session {sid} | {model}")
            print(f"{CLR['dim']}    cwd: {cwd}{CLR['reset']}")
        elif state.args.verbose and st in ("hook_started", "hook_response"):
            print(f"{CLR['dim']}  . {st} {event.get('hook_name', '?')} exit={event.get('exit_code', '')}{CLR['reset']}")

    elif t == "assistant":
        msg = event.get("message", {})
        msg_id = msg.get("id")
        # Multiple stream events share an msg_id (one per content block).
        # Bump the turn counter only when msg_id changes; print a header on
        # first sight; subsequent events for the same msg just emit content.
        if msg_id != state.last_assistant_msg_id:
            state.turn += 1
            state.last_assistant_msg_id = msg_id
            u = msg.get("usage") or {}
            state.cum_in += u.get("input_tokens", 0) or 0
            state.cum_out += u.get("output_tokens", 0) or 0
            state.cum_cache_read += u.get("cache_read_input_tokens", 0) or 0
            ts = datetime.now().strftime("%H:%M:%S")
            label = f"{ts}  turn {state.turn}"
            if not state.args.no_tokens:
                label += (
                    f"  in={_fmt_tokens(state.cum_in)}"
                    f"  cache={_fmt_tokens(state.cum_cache_read)}"
                    f"  out={_fmt_tokens(state.cum_out)}"
                )
            _hr(state, label)
        for c in msg.get("content", []):
            ct = c.get("type")
            if ct == "text":
                text = (c.get("text") or "").strip()
                if text:
                    wrapped = _wrap(text, state.term_width - 2, "  ", "  ")
                    print(f"{CLR['bold']}{wrapped}{CLR['reset']}")
            elif ct == "tool_use":
                if state.args.text_only:
                    continue
                name = c.get("name", "?")
                inp = c.get("input", {}) or {}
                label, detail = _fmt_tool(name, inp)
                arrow = f"{CLR['magenta']}->{CLR['reset']}"
                head = f"  {arrow} {CLR['cyan']}{label}{CLR['reset']}"
                if not detail:
                    print(head)
                    continue
                detail_str = str(detail).rstrip()
                # Soft cap on raw tool input (cmd/content snippet) length
                if len(detail_str) > 800:
                    detail_str = detail_str[:800] + f"... ({len(detail_str) - 800} more chars)"
                if "\n" in detail_str:
                    print(head)
                    for ln in detail_str.split("\n"):
                        print(f"     {CLR['gray']}{ln}{CLR['reset']}")
                else:
                    # short single-line: inline with colon
                    if len(head) + len(detail_str) + 2 < state.term_width:
                        print(f"{head}: {CLR['gray']}{detail_str}{CLR['reset']}")
                    else:
                        print(head)
                        print(f"     {CLR['gray']}{detail_str}{CLR['reset']}")

    elif t == "user":
        if state.args.text_only:
            return
        msg = event.get("message", {})
        for c in msg.get("content", []):
            if c.get("type") != "tool_result":
                continue
            body = _flatten_tool_result_content(c.get("content", "")).strip()
            is_error = bool(c.get("is_error"))
            arrow_color = CLR["red"] if is_error else CLR["green"]
            arrow = f"{arrow_color}<-{CLR['reset']}"
            body_color = CLR["red"] if is_error else CLR["dim"]
            if not body:
                print(f"  {arrow} {CLR['dim']}(no output){CLR['reset']}")
                continue
            truncated, extra_lines = _truncate_lines(
                body, state.args.max_result_lines, max(40, state.term_width - 6)
            )
            print(f"  {arrow}")
            for ln in truncated.split("\n"):
                print(f"     {body_color}{ln}{CLR['reset']}")
            if extra_lines:
                print(f"     {CLR['dim']}...({extra_lines} more lines){CLR['reset']}")

    elif t == "result":
        ok = event.get("subtype") == "success"
        cost = event.get("total_cost_usd", 0) or 0
        dur = (event.get("duration_ms") or 0) / 1000
        color = CLR["yellow"] if ok else CLR["red"]
        _hr(state, f"finished | ok={ok} | {dur:.0f}s | ${cost:.2f}")
        if not ok:
            err = event.get("error") or event.get("subtype") or "unknown"
            print(f"{color}    error: {err}{CLR['reset']}")

    elif t == "rate_limit_event":
        if state.args.verbose:
            info = event.get("rate_limit_info") or {}
            print(f"{CLR['dim']}  . rate_limit status={info.get('status')} overage={info.get('overageStatus')}{CLR['reset']}")

    else:
        if state.args.verbose:
            print(f"{CLR['dim']}  . [{t}]{CLR['reset']}")


def stream_file(path: Path, state: State) -> None:
    """tail -F equivalent: keep reading; handle truncation/rotation via inode tracking."""
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
                continue
            render(ev, state)
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


def stream_stdin(state: State) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        render(ev, state)
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live human-readable view of a Claude Code stream-json log.")
    ap.add_argument("input", help="path to .jsonl log, or '-' for stdin")
    ap.add_argument("-v", "--verbose", action="store_true", help="include hooks + rate-limit + unknown events")
    ap.add_argument("-t", "--text-only", action="store_true", help="show assistant text only; hide tool calls + results")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--color", action="store_true")
    ap.add_argument("--no-tokens", action="store_true", help="omit cumulative token counts from turn header")
    ap.add_argument("--max-result-lines", type=int, default=8, help="max lines of tool result to show (default 8)")
    ap.add_argument("--width", type=int, default=0, help="override terminal width (0=auto-detect)")
    args = ap.parse_args()

    if args.no_color:
        enabled = False
    elif args.color:
        enabled = True
    else:
        enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    _setup_colors(enabled)

    state = State(args)
    if args.input == "-":
        stream_stdin(state)
    else:
        stream_file(Path(args.input), state)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
