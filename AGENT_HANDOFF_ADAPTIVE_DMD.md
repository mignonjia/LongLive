# Agent Handoff: LongLive DMD + Adaptive Streaming

Date: 2026-04-25

This document summarizes the local changes, debugging conclusions, and expected
runtime behavior from the recent LongLive DMD/adaptive training work.

## Main Goal

Use plain `distribution_loss: dmd` for streaming training, then enable adaptive
sequence length training without the previous KV/cache crash and without always
training the generator on the same chunk positions.

## Important Files

- `model/dmd.py`
- `trainer/distillation.py`
- `configs/longlive_train_long.yaml`
- `configs/longlive_train_adaptive.yaml`
- `train_long.sh`
- `train_adaptive.sh`

## DMD Streaming Fix

Problem observed earlier:

- Plain `dmd` with `streaming_training: true` crashed during long streaming
  training.
- The crash happened after several chunks, not immediately.
- The old path effectively used the wrong inference/training pipeline for DMD
  streaming state.

Fix:

- `model/dmd.py` now imports and uses `StreamingTrainingPipeline` when
  `args.streaming_training` is true.
- This makes plain `DMD` use the same streaming pipeline semantics needed by
  the chunked KV-cache training path.

Expected confirmation in logs:

```text
[DMDDiag] initialized StreamingTrainingPipeline for streaming_training=True ...
streaming training enabled: chunk_size=21, max_length=129
```

Note: temporary noisy DMD diagnostic prints were later removed. If current logs
do not show `[DMDDiag]`, that is expected after cleanup.

## Stable Long DMD Config

`configs/longlive_train_long.yaml` is the stable non-adaptive long-training
baseline.

Important settings:

```yaml
distribution_loss: dmd
global_sink: true
streaming_training: true
streaming_chunk_size: 21
streaming_max_length: 129
streaming_min_new_frame: 18
dfake_gen_update_ratio: 5
```

Interpretation:

- Chunk size is 21 frames.
- After the first chunk, overlap is 3 frames because `21 - 18 = 3`.
- Generator update ratio 5 means the generator trains every 5 global steps.
- That gives 4 critic-only steps between generator updates.

Known good command:

```bash
nohup bash train_long.sh 2>&1 | tee logs_long_dmd_diag.out
```

This command was reported to work well after the DMD streaming fix.

## Adaptive Training Changes

`configs/longlive_train_adaptive.yaml` now uses DMD and staged max lengths.

Current intended schedule:

```yaml
max_iters: 3500
distribution_loss: dmd
streaming_training: true
streaming_chunk_size: 21
streaming_max_length: 129
streaming_min_new_frame: 18
adaptive_debug_log: false
streaming_max_length_schedule:
  - start_step: 0
    end_step: 400
    max_length: 39
  - start_step: 400
    end_step: 800
    max_length: 57
  - start_step: 800
    end_step: 1200
    max_length: 75
  - start_step: 1200
    end_step: 1600
    max_length: 93
  - start_step: 1600
    end_step: 2000
    max_length: 111
  - start_step: 2000
    end_step: 3500
    max_length: 129
```

Reasoning:

- Skip the 21-frame stage because the separate init stage already covers it.
- Start adaptive long training at 39 frames.
- Spend 400 steps per intermediate stage.
- End at 129 frames and continue there until step 3500.

Expected first adaptive log sign:

```text
[StreamingTrain-Model] current_training_chunk: (0 -> 21)/39
```

That means the schedule is active and the first adaptive max length is 39.

## Random Generator Gap Logic

Motivation:

- Fixed generator update cadence can repeatedly train the generator on the same
  chunk positions.
- The adaptive run should randomize generator update positions while preserving
  approximately the same average update frequency as the stable long config.

Config:

```yaml
generator_update_gap_choices:
- 3
- 4
- 5
```

Semantics:

- After a generator update, the trainer samples one value from `[3, 4, 5]`.
- The sampled value is the number of critic-only steps before the next generator
  update.
- The average gap is 4 critic-only steps, matching `dfake_gen_update_ratio: 5`
  in `longlive_train_long.yaml`.

Implementation detail:

- `_update_next_generator_step()` sets:

```python
self.next_generator_step = int(self.step) + selected_gap
```

- This intentionally does not add `+ 1`.
- At the point `_update_next_generator_step()` is called, `self.step` has
  already advanced after the generator update.
- The previous `+ 1` behavior made the real gap one step larger than the sampled
  value.

Distributed behavior:

- Rank 0 samples the gap.
- The selected gap is broadcast to all ranks with `dist.broadcast`.
- This keeps every rank aligned on the same `next_generator_step`.

## Low-Volume Gap Logging

`trainer/distillation.py` now prints a normal, low-volume line every time a new
adaptive generator gap is sampled:

```text
[AdaptiveGap] step=<step> sampled_gap=<gap> next_generator_step=<step>
```

This prints even when:

```yaml
adaptive_debug_log: false
```

It only prints when the next generator gap is sampled, not on every layer,
chunk, or training loop iteration.

Important:

- A run already started before this code change will not show `[AdaptiveGap]`.
- Restart the run to load the new Python code.

## Optional Debug Logging

`adaptive_debug_log: true` enables additional `[AdaptiveDiag]` lines:

- scheduler enabled message,
- selected max length stage,
- train-generator decision per iteration,
- current streaming length and next generator step.

This was useful for confirming behavior but should remain false for normal
training because it is more verbose than needed.

## Expected Run Command

For the adaptive run:

```bash
nohup bash train_adaptive.sh > logs_long_adaptive.out 2>&1 &
```

`train_adaptive.sh` currently runs:

```bash
torchrun \
  --nproc_per_node=8 \
  --master_port=29508 \
  train.py \
  --config_path configs/longlive_train_adaptive.yaml \
  --logdir logs_long_adaptive \
  --wandb-save-dir wandb \
  --run-name longlive_adaptive \
  --no-one-logger
```

## What To Check In Logs

After restart with the current code, check:

```bash
grep -E "AdaptiveGap|current_training_chunk|Traceback|RuntimeError|CheckpointError" logs_long_adaptive.out
```

Desired signs:

- First stage starts at `/39`, not `/21`.
- `[AdaptiveGap]` appears after generator updates.
- `sampled_gap` varies among 3, 4, and 5 over time.
- `next_generator_step` changes after each generator update.
- No `Traceback`, `RuntimeError`, or checkpoint/cache shape error.

Example desired pattern:

```text
[StreamingTrain-Model] current_training_chunk: (0 -> 21)/39
[AdaptiveGap] step=1 sampled_gap=4 next_generator_step=5
[AdaptiveGap] step=6 sampled_gap=3 next_generator_step=9
[AdaptiveGap] step=10 sampled_gap=5 next_generator_step=15
```

Exact numbers will vary because the gap is random.

## Notes And Cautions

- `next_generator_step` is expected to remain constant between generator
  updates. It only changes when a new gap is sampled after a generator update.
- `global_sink: true` helped the stable long DMD run. The adaptive config should
  be checked if changing sink behavior.
- Do not use broad rsync with `--delete` for logs or generated directories. A
  prior broad log pull removed some local generated directories that were absent
  remotely. Prefer single-file log pulls.
- The remote host used successfully in the recent session was:

```text
m2-login-003.mbzuai-hpc.teleport.sh
```

- Remote workspace path:

```text
~/mhuo/FastVideo/
```

## Current Mental Model

The original crash was not solved by changing latent length alone. The key
functional fix was making plain DMD use the correct streaming training pipeline.
After that, the stable long DMD run worked. The adaptive method should now be
safe under the same DMD streaming path, with the remaining concern being whether
the staged max lengths and randomized generator gap produce the intended update
coverage. The `[AdaptiveGap]` log is the lightweight verification hook for that.
