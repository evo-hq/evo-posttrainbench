"""Modal wrapper for evo-posttrainbench.

Runs the same PostTrainBench AIME flow on Modal's H100 + a persistent Volume.
Three entrypoints:

    modal run    scripts/modal_app.py::dry_run                            # ~few min, cheap pipeline check
    modal run    scripts/modal_app.py::train --model Qwen/Qwen3-4B-Base   # real 10h training
    modal deploy scripts/modal_app.py                                     # leave dashboard up (web URL)

One-time setup (creates Modal Secrets):
    # We use Claude Code OAuth (Max subscription) -- run `claude setup-token` locally first.
    modal secret create anthropic CLAUDE_CODE_OAUTH_TOKEN=...
    modal secret create hf        HF_TOKEN=...               # gemma-3-4b-pt is gated
    modal secret create wandb     WANDB_API_KEY=...          # optional

The container is the same H100 the JarvisLabs path uses; everything under
/workspace (the Volume) persists across runs the way /home does on JarvisLabs.
"""
from __future__ import annotations

import os
import subprocess

import modal

APP = "evo-posttrainbench"
EVO_BRANCH = "feat/model-update"

# Image: PostTrainBench's pinned starting env (matches containers/requirements-direct.txt
# + their .def), plus Claude Code 2.1.76 and evo from our branch. Layer order goes
# stable -> volatile so the evo branch update doesn't bust the heavy ML layers.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "curl", "build-essential", "tmux", "tree")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code@2.1.76",
        "pip install --no-cache-dir uv",
    )
    # This fork: clone for containers/requirements-direct.txt + scripts/run.sh
    # + agents/claude_evo_max + src/eval/*.
    .run_commands("git clone https://github.com/evo-hq/evo-posttrainbench.git /opt/ptb")
    # PostTrainBench's pinned starting env + vLLM + flash-attn
    # `--torch-backend=auto` fails during a Modal image build because there's no
    # GPU at build time for uv to detect; pin to cu128 (matches our 12.9.1 base).
    .run_commands(
        "uv pip install --system --no-cache vllm==0.11.0 --torch-backend=cu128",
    )
    .run_commands(
        "uv pip install --system --no-cache -r /opt/ptb/containers/requirements-direct.txt",
    )
    # trackio: wandb-API-compatible OSS tracker; logs to a HF Space.
    # Pin <0.10 -- trackio 0.10+ requires gradio 6 + huggingface-hub>=1.0, which
    # conflicts with PostTrainBench's pinned transformers 4.57.3 (needs hf-hub<1.0).
    # 0.4.0-0.9.0 explicitly declare huggingface-hub<1.0.0 and gradio 5.x.
    .run_commands("uv pip install --system --no-cache 'trackio<0.10'")
    # flash-attn doesn't declare wheel as a build dep, so --no-build-isolation
    # fails without wheel pre-installed.
    .run_commands(
        "uv pip install --system --no-cache wheel setuptools",
        "uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation",
    )
    # AIME eval deps: inspect_evals registers the aime2025 task; the vllm-stdout
    # fork is what evaluate.py uses for vLLM-backed inspect runs.
    .run_commands(
        "git clone https://github.com/UKGovernmentBEIS/inspect_evals.git /opt/inspect_evals "
        "&& cd /opt/inspect_evals && git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 "
        "&& uv pip install --system --no-cache .",
        "git clone https://github.com/rank-and-file/inspect_ai_vllm_stdout.git "
        "/opt/inspect_ai_vllm_stdout && cd /opt/inspect_ai_vllm_stdout "
        "&& uv pip install --system --no-cache .",
    )
    # evo from our branch -- last so flipping the branch only re-runs this layer.
    # Fetch the evo-hook-drain binary from GitHub's /releases/latest/ redirect.
    # That URL always points at the LATEST STABLE release's asset, no API call
    # or jq needed. The binary's wire protocol is stable enough across minor
    # versions that an older-stable binary works fine with a newer/alpha CLI.
    # If the fetch 404s (no release has the asset for our arch), the env var
    # is set to a non-existent path -- ensure_hook_drain_binary falls back to
    # its own fetch logic at install, and if that fails too, evo direct just
    # doesn't work for this run (train function continues fine without it).
    .run_commands(
        f"git clone -b {EVO_BRANCH} https://github.com/evo-hq/evo.git /opt/evo "
        "&& uv tool install --editable /opt/evo/plugins/evo "
        "&& (curl -fsSL -o /opt/evo-hook-drain "
        "    https://github.com/evo-hq/evo/releases/latest/download/evo-hook-drain-linux-amd64 "
        "    && chmod +x /opt/evo-hook-drain "
        "    && echo '[hook-drain] staged from /releases/latest/download/') "
        "   || echo '[hook-drain] WARN: latest-release fetch failed; evo direct may not work'",
    )
    .env({
        "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
        # ensure_hook_drain_binary checks this env var first -- bypasses its
        # own per-version fetch (which 404s for pre-publish alphas) by copying
        # the binary we already staged from the latest available release.
        "EVO_HOOK_DRAIN_BINARY": "/opt/evo-hook-drain",
    })
)

app = modal.App(APP, image=image)

# v2 Volume scales better for the writer+dashboard-reader pattern.
vol = modal.Volume.from_name(f"{APP}-runs", create_if_missing=True, version=2)

SECRETS = [
    modal.Secret.from_name("anthropic"),   # CLAUDE_CODE_OAUTH_TOKEN (Max subscription)
    modal.Secret.from_name("hf"),          # HF_TOKEN (accept gemma-3-4b-pt license on HF first)
]
# Optional: for W&B training curves, create the secret and add it here:
#   modal secret create wandb WANDB_API_KEY=...
#   SECRETS.append(modal.Secret.from_name("wandb"))

COMMON = dict(volumes={"/workspace": vol}, secrets=SECRETS)


# Refresh CLI from /opt/evo at container start so the version on PATH matches
# the marketplace plugin (which `evo install` always pulls fresh from main).
# The image layer is cached at first-build state, so without this the CLI
# drifts behind -- breaking both the discover skill's version-match gate
# (train function) and the EVO_DASHBOARD_HOST=0.0.0.0 binding (dashboard
# function, where older CLI ignored the env var and bound 127.0.0.1, causing
# Modal's wait_for_web_server to time out). Used by both train and dashboard.
# The branch gets force-pushed so we hard-reset rather than --ff-only.
_REFRESH_EVO_CLI = (
    f"cd /opt/evo && git fetch origin {EVO_BRANCH} && "
    f"git reset --hard origin/{EVO_BRANCH} && "
    "uv tool install --reinstall --editable /opt/evo/plugins/evo; "
    # The evo-hook-drain binary is staged at image build via a GitHub release
    # fetch (latest tag with the asset for our platform). The binary is
    # wire-compatible across minor versions, so we don't re-fetch on every
    # CLI refresh -- only when the image is rebuilt.
)


def _agent_cmd(task: str, model: str, hours: int) -> str:
    """The bash one-liner that installs the plugin into the volume's CLAUDE_CONFIG_DIR
    (idempotent) and runs the same scripts/run.sh used on JarvisLabs."""
    return (
        "set -euo pipefail; "
        "cd /opt/ptb && git pull --ff-only origin main; "                  # always run the latest scripts
        + _REFRESH_EVO_CLI +
        "export WORK=/workspace REPO=/opt/ptb "
        "HF_HOME=/workspace/hf CLAUDE_CONFIG_DIR=/workspace/.claude "
        "EVO_DASHBOARD_HOST=0.0.0.0 EVO_DASHBOARD_PORT=8080 "
        "IS_SANDBOX=1 "                                                    # claude --dangerously-skip-permissions otherwise rejects root
        'TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-alok97/posttrain-runs}"; '
        "mkdir -p \"$HF_HOME\" \"$CLAUDE_CONFIG_DIR\"; "
        # Install plugin from the LOCAL /opt/evo clone (feat/model-update tip),
        # NOT from the public marketplace -- marketplace points at origin/main
        # which lags behind feat/model-update. From-path uses the same source
        # the CLI was built from, so skills + CLI versions stay in sync.
        # ensure_hook_drain_binary reads EVO_HOOK_DRAIN_BINARY (set in image
        # env to /opt/evo-hook-drain, rebuilt at runtime via _REFRESH_EVO_CLI)
        # to bypass the GitHub release fetch (which 404s for pre-publish alphas).
        "evo install claude-code --from-path /opt/evo; "
        f"cd \"$REPO\" && bash scripts/run.sh run {task} {model} {hours}"
    )


@app.function(gpu="H100!", timeout=12 * 3600, **COMMON)
def train(
    model: str = "Qwen/Qwen3-4B-Base",
    hours: int = 10,
    task: str = "aime2025",
):
    """Real run: the agent post-trains the base model on the task. ~10h on H100.

    Modal wall-clock timeout is 12h (vs. the 10h agent budget) to absorb
    container startup (~3min) + final eval (~5-10min for AIME 2025) plus
    headroom. The agent's own timer.sh starts at solve.sh launch (run.sh:100)
    so the 10h budget is wall-clock from when Claude actually starts, not
    from container boot.

    `gpu="H100!"` pins H100 (without it Modal silently upgrades to H200, which
    changes pricing and may break flash-attn/vLLM pinned kernels).
    """
    subprocess.run(["bash", "-lc", _agent_cmd(task, model, hours)], check=True)
    vol.commit()


@app.function(gpu="H100!", timeout=15 * 60, **COMMON)
def dry_run():
    """Cheap pipeline check (~few min): GPU + deps + volume + secrets + CLIs.
    Run this before paying for a 10h train()."""
    print("=== GPU ===", flush=True)
    subprocess.run(["nvidia-smi", "-L"], check=True)

    print("=== Python deps ===", flush=True)
    import torch                       # noqa: F401
    import vllm                        # noqa: F401
    import trl                         # noqa: F401
    import peft                        # noqa: F401
    import inspect_evals.aime2025      # noqa: F401
    print(f"ok: {torch.cuda.get_device_name(0)}", flush=True)

    print("=== Volume write/read ===", flush=True)
    p = "/workspace/dry_run.txt"
    with open(p, "w") as f:
        f.write("hello from dry_run\n")
    vol.commit()
    with open(p) as f:
        print(f"wrote {p}: {f.read().strip()}", flush=True)

    print("=== Auth env ===", flush=True)
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        raise SystemExit("ERROR: CLAUDE_CODE_OAUTH_TOKEN missing -- the experiment uses Claude Code OAuth (Max subscription). "
                         "Run `claude setup-token` locally and add it to the `anthropic` Modal secret.")
    for k in ("CLAUDE_CODE_OAUTH_TOKEN", "HF_TOKEN"):
        print(f"  {k}: {'set' if os.environ.get(k) else 'MISSING'}", flush=True)

    print("=== CLIs ===", flush=True)
    subprocess.run(["claude", "--version"], check=True)
    subprocess.run(["evo", "--version"], check=True)

    print("\nALL OK -- safe to invoke train()", flush=True)


@app.function(timeout=120, **COMMON)
def evo_direct(message: str):
    """Inject an `evo direct` user-authoritative directive into the running
    agent's session. Wraps the message in the [EVO DIRECTIVE] banner via the
    inject mechanism; the running agent sees it as a new user turn on its
    next hook fire.

    Usage: `modal run scripts/modal_app.py::evo_direct --message "<text>"`
    """
    import shlex
    import subprocess
    quoted = shlex.quote(message)
    subprocess.run(["bash", "-c", f"""
        set -eu
        LATEST=$(ls -1dt /workspace/runs/*/task 2>/dev/null | head -1 || true)
        if [ -z "$LATEST" ] || [ ! -d "$LATEST/.evo" ]; then
            echo "no run with .evo/ under /workspace/runs/*/task"
            exit 1
        fi
        cd "$LATEST"
        echo "broadcasting directive into $LATEST/.evo/"
        evo direct {quoted}
    """], check=True)
    vol.commit()


@app.function(timeout=10 * 60, **COMMON)
def directive_smoke(token: str = "PINEAPPLE_42"):
    """End-to-end smoke for `evo direct` delivery. ~3 min, no GPU.

    Flow:
      1. Refresh CLI (same as train), install plugin into volume.
      2. Stage hook-drain binary (same fallback as _agent_cmd).
      3. Create a fresh workspace with `evo init`.
      4. Spawn a background bash that sleeps 30s, then runs
         `evo direct "<token>"` into the workspace.
      5. Start `claude --print` with a prompt that runs `bash sleep 60`,
         then `bash sleep 60` again (separate tool calls). Between the
         two sleeps, the drain hook fires and should deliver the directive.
      6. After claude finishes, grep its output for the token.
      7. Exit 0 if token present, 1 otherwise.

    Usage: `modal run scripts/modal_app.py::directive_smoke`
    Optional: `--token MY_SECRET` (defaults to PINEAPPLE_42).
    """
    import shlex
    import subprocess
    qtoken = shlex.quote(token)
    rc = subprocess.run(["bash", "-lc", f"""
        set -eu
        export WORK=/workspace REPO=/opt/ptb
        export HF_HOME=/workspace/hf
        export CLAUDE_CONFIG_DIR=/workspace/.claude.smoke
        export IS_SANDBOX=1
        mkdir -p "$HF_HOME" "$CLAUDE_CONFIG_DIR"

        # 1. CLI refresh (same as train)
        cd /opt/evo && git fetch origin {EVO_BRANCH} && git reset --hard origin/{EVO_BRANCH}
        uv tool install --reinstall --editable /opt/evo/plugins/evo

        # 2. Plugin install + hook-drain fallback (same as _agent_cmd)
        evo install claude-code
        PLUGIN_VER_DIR=$(ls -1dt "$CLAUDE_CONFIG_DIR"/plugins/cache/evo-hq-evo/evo/*/ 2>/dev/null | head -1)
        if [ -n "$PLUGIN_VER_DIR" ] && [ ! -x "$PLUGIN_VER_DIR/bin/evo-hook-drain" ]; then
            VER=$(basename "${{PLUGIN_VER_DIR%/}}")
            ARCH=$(uname -m); case "$ARCH" in x86_64|amd64) T=linux-amd64;; aarch64|arm64) T=linux-arm64;; *) T="";; esac
            if [ -n "$T" ]; then
                mkdir -p "$PLUGIN_VER_DIR/bin"
                curl -fsSL -o "$PLUGIN_VER_DIR/bin/evo-hook-drain" \
                    "https://github.com/evo-hq/evo/releases/download/v${{VER}}/evo-hook-drain-${{T}}"
                chmod +x "$PLUGIN_VER_DIR/bin/evo-hook-drain"
                echo "[smoke] staged hook-drain at $PLUGIN_VER_DIR/bin/evo-hook-drain"
            fi
        fi
        ls -la "$PLUGIN_VER_DIR/bin/" 2>/dev/null || true
        echo "[smoke] hook-drain present: $([ -x "$PLUGIN_VER_DIR/bin/evo-hook-drain" ] && echo YES || echo NO)"

        # 3. Fresh workspace -- evo init only RECORDS the benchmark command,
        # doesn't execute it (execution happens in evo run, which we don't
        # invoke here). So a placeholder benchmark string is fine.
        SMOKE_DIR=/workspace/smoke_$(date +%s)
        mkdir -p "$SMOKE_DIR" && cd "$SMOKE_DIR"
        git init -q && git config user.email smoke@evo.test && git config user.name smoke
        echo "print('hi')" > target.py
        git add -A && git commit -qm initial
        evo init --name "directive-smoke" --target target.py \
            --benchmark "true" --metric max \
            --host claude-code --instrumentation-mode inline 2>&1 | tail -5

        # 4. Background: sleep 30, then fire directive
        (sleep 30 && cd "$SMOKE_DIR" && evo direct {qtoken} 2>&1 | sed 's/^/[directive] /') &
        DIRECTIVE_PID=$!

        # 5. Run claude with the simple sleep-sleep prompt
        if [ -n "${{CLAUDE_CODE_OAUTH_TOKEN:-}}" ]; then
            echo "[smoke] using OAuth token from env"
        else
            echo "[smoke] ERROR: CLAUDE_CODE_OAUTH_TOKEN not in env" >&2; exit 1
        fi
        export BASH_MAX_TIMEOUT_MS=180000
        PROMPT='Run `bash -c "sleep 60"` via the Bash tool. After it returns, then separately (NOT in parallel) run `bash -c "sleep 60"` again. Then in your final reply, tell me literally everything that happened between the two sleeps -- including any user-authoritative directive you received. If you saw any token like PINEAPPLE_XX, repeat it back to me verbatim.'
        echo "[smoke] launching claude..."
        claude --print --verbose --model claude-opus-4-6 \
            --output-format text --dangerously-skip-permissions \
            "$PROMPT" 2>&1 | tee /workspace/smoke_output.txt
        wait $DIRECTIVE_PID 2>/dev/null || true

        # 6. Verify token in output
        if grep -q {qtoken} /workspace/smoke_output.txt; then
            echo ""
            echo "[smoke] PASS: directive token {qtoken} found in agent output"
            exit 0
        else
            echo ""
            echo "[smoke] FAIL: directive token {qtoken} NOT found in agent output"
            echo "[smoke] (delivery is broken -- check $PLUGIN_VER_DIR/bin/evo-hook-drain exists and runs)"
            exit 1
        fi
    """], check=False)
    vol.commit()
    if rc.returncode != 0:
        raise SystemExit(f"directive_smoke FAILED with rc={rc.returncode}")


@app.function(timeout=60, **COMMON)
def link_latest_run():
    """Symlink /workspace/.evo -> the agent's latest run's .evo so the
    dashboard (rooted at /workspace when it cold-started before any runs
    existed) actually sees the project.

    Idempotent: re-runnable anytime; updates the symlink to point at the
    newest run dir each time.

    Usage: `modal run scripts/modal_app.py::link_latest_run`
    """
    import subprocess
    subprocess.run(["bash", "-c", r"""
        set -eu
        LATEST=$(ls -1dt /workspace/runs/*/task 2>/dev/null | head -1 || true)
        if [ -z "$LATEST" ] || [ ! -d "$LATEST/.evo" ]; then
            echo "no run with .evo/ found under /workspace/runs/*/task -- nothing to link"
            exit 1
        fi
        ln -snf "$LATEST/.evo" /workspace/.evo
        echo "linked /workspace/.evo -> $LATEST/.evo"
        ls -la /workspace/.evo
    """], check=True)
    vol.commit()


@app.function(scaledown_window=1200, **COMMON)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8080, startup_timeout=180)
def dashboard():
    """Public HTTPS dashboard against the latest run dir on the Volume.
    URL: https://<workspace>--<app>-dashboard.modal.run

    startup_timeout=180 to absorb the CLI refresh (~30s) + Flask startup.
    Uses `evo-dashboard` (the direct entry from pyproject -- evo.dashboard:main)
    instead of `evo dashboard` (which goes through a supervisor + DEVNULL'd
    subprocesses, defeating env-var debugging). evo-dashboard reads
    EVO_DASHBOARD_HOST/PORT directly and binds Flask once.
    """
    cmd = (
        "set -eu; "                                                         # no pipefail: LATEST=$(ls|head) trips it
        + _REFRESH_EVO_CLI +
        "LATEST=$(ls -1dt /workspace/runs/*/task 2>/dev/null | head -1 || true); "
        ': "${LATEST:=/workspace}"; '                                       # POSIX default-assignment
        'cd "$LATEST" && '
        "EVO_DASHBOARD_HOST=0.0.0.0 EVO_DASHBOARD_PORT=8080 exec evo-dashboard"
    )
    subprocess.Popen(["bash", "-lc", cmd])

    # Modal v2 volumes are not strictly read-after-write consistent across
    # containers -- the dashboard's mount serves cached file content from
    # before the train container's writes, so /api/graph etc. show stale
    # state. evo-dashboard's load_graph() re-reads per request but Python's
    # open() returns the cached bytes. vol.reload() flushes the mount cache.
    # Background daemon polls every 5s while the server runs.
    import threading
    import time as _time
    def _reload_loop():
        while True:
            _time.sleep(5)
            try:
                vol.reload()
            except Exception:
                pass
    threading.Thread(target=_reload_loop, daemon=True).start()
