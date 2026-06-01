"""Bridge TRL/transformers Trainer logs into trackio.

Trackio is wandb-API-compatible (`import trackio as wandb`) but does NOT
install itself as the `wandb` module in sys.modules. As a result, setting
`report_to="wandb"` in TrainingArguments either fails (no wandb installed)
or imports real wandb if anything else pulled it in -- neither path lands
in trackio, and your HF Space stays empty for the whole run while training
loss falls silently.

The fix is a small TrainerCallback that forwards TRL's `on_log` event to
`trackio.log()`. Register it on the trainer; leave `report_to="none"` in
TrainingArguments.

Usage:

    from scripts.trl_trackio_callback import TrackioCallback

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=SFTConfig(..., report_to="none"),   # important: not "wandb"
        processing_class=tokenizer,
    )
    trainer.add_callback(TrackioCallback(project="ptb", run_name="sft_v2"))
    trainer.train()
"""

from __future__ import annotations

import os

from transformers import TrainerCallback


class TrackioCallback(TrainerCallback):
    """Forward Trainer log events to trackio. One init per Trainer; finishes on train_end."""

    def __init__(
        self,
        project: str = "ptb",
        run_name: str | None = None,
        space_id: str | None = None,
    ):
        import trackio  # imported lazily so module-import doesn't fail without trackio

        space_id = space_id or os.environ.get("TRACKIO_SPACE_ID", "")
        if not space_id:
            raise RuntimeError(
                "TRACKIO_SPACE_ID env var unset and no space_id passed. "
                "Set TRACKIO_SPACE_ID (e.g. 'alok97/posttrain-runs') so trackio "
                "knows which HF Space to log to."
            )
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError(
                "HF_TOKEN env var unset. trackio needs a write-scope token to push "
                "metrics to the HF Space. Add HF_TOKEN to your Modal `hf` secret."
            )

        trackio.init(project=project, name=run_name, space_id=space_id)
        self._trackio = trackio

        # Bridge to evo dashboard: write the trackio Space URL into the
        # experiment's traces dir so the dashboard can render a link + scrape
        # the corresponding parquet for the live curve. EVO_TRACES_DIR is
        # exported by `evo run` for the activity it spawned; absent if the
        # user ran this script outside an evo experiment, which is fine.
        traces_dir = os.environ.get("EVO_TRACES_DIR")
        if traces_dir:
            try:
                os.makedirs(traces_dir, exist_ok=True)
                url = f"https://huggingface.co/spaces/{space_id}"
                # run_name lets the dashboard scope parquet rows; project tells
                # it which parquet file inside the dataset.
                payload = (
                    f"url={url}\n"
                    f"space_id={space_id}\n"
                    f"project={project}\n"
                    f"run_name={run_name or ''}\n"
                )
                with open(os.path.join(traces_dir, ".trackio_url"), "w") as f:
                    f.write(payload)
            except Exception:
                # Dashboard surfacing is observability, not correctness. Never
                # let a write failure here kill training.
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Trainer's on_log fires for train and eval logs; forward both.
        self._trackio.log(logs, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._trackio.finish()
        except Exception:
            pass
