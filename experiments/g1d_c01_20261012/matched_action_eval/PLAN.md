# Plan

User: inspect 992446 migration and A100 full FT100k/A800 LoRA100k, compare actions on identical observations with pi0.5 baseline; compute metrics and plot comparison/open-loop curves.

Attach existing baseline7500 fixed 200 heldout anchors. Use baseline H32 first30, C01 H30/native16; identical official split, quantile state transform, image preprocessing, prompts, anchor order, 10 denoise steps and seed protocol. Verify indices/tasks/targets/masks; export current-observation digests for audit where available. Existing baseline already accepted in prior comparisons; reuse existing predictions, do not retrain.

A800 LoRA100k completed, select97500 minimum heldout composite loss. A100 training still60k/100k; select committed60000 as interim comparator, preserve ongoing training. If no safe concurrent memory for FP32 parameter inference, record failure and use explicit alternative; never silently change precision or protocol.

Outputs: raw prediction NPZ and evaluation JSON, canonical metric_contract.json, pooled and per-joint/per-task/horizon/state-anchor and temporal metrics, source-episode bootstrap paired intervals, training curves, same-observation open-loop trajectories and reference images, PNG+PDF figures inspected before completion, Chinese report and STATUS update.

Limitations: selected heldout is not test; only 200 teacher-observation chunks, not closed-loop rollout; unequal update/sample/LR budgets; action and effect latent noise dimensionalities differ; pooled16D mixes arm radians with gripper units, report groups separately. Smoothness proxies are predicted-chunk diagnostics, not measured robot vibration.

Tool availability deviation: skill bash_exec/artifact/memory tools are unavailable (metadata inspected); use available exec_command and durable local reports instead. No new Python environment required; reuse installed compatible runtimes.

## Input-audit refinement

Baseline7500 source directory on A800 has historical predictions/provenance, not params. A baseline weight rerun there is unavailable. Attach original NPZ with SHA2565cb9ca54a89c91d20f6daba2c5cdd3f2814d8973547aab51d86ebdff4b908ab5 only after reconstructing original input preprocessing at all200 anchors. Exact raw RGB pixel/state/token/mask identity is required. Normalized float images differ by<=2e-7 (JAX reciprocal vs NumPy division rounding); preserve original tensor hashes and report this difference rather than claiming all normalized tensor bits equal. C01 LoRA/FT input tensors must match bitwise.

Four short subtask segments produce34 padded future steps out of6000. Exclude them from all position, derivative, distribution and bootstrap metrics. Plot padded regions explicitly.

## Runtime verification

FullFT60000 checkpoint-native inference succeeded on rank1 alongside training, parameter bytes13,432,789,248; peak combined GPU use77,596MiB within80GiB. No precision reduction or training interruption was needed. Smoke2 then main200 completed. Baseline audit all200 passed raw-pixel/nonimage identity; maximum normalized-image difference1.1920928955078125e-7. Retained strict target/mask checks.
