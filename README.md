# Continual AI Lab v0.4.1 — Multi-agent verified learning + gated QLoRA

**Status:** runnable native-Windows local multi-agent loop; scheduled CPU-only GitHub experiment; optional real CUDA QLoRA with independent two-split tests and automatic *local pointer* promotion. This does **not** establish general intelligence improvement and is not a 24/7 hosted inference API.

## The four agents

| Role | What it does | Per-cycle LLM calls |
| --- | --- | ---: |
| Learner | Answers a generated arithmetic task and retains independently labelled experience | 1 |
| Optimiser | Tests an alternate prompt strategy; UCB algorithm uses measured successes to select future strategies | 1 |
| Independent verifier | Checks at least TWO fresh reserved tasks; adds bounded checks until verification inference time reaches at least half the cycle | 2–10 |
| Resource governor | Pauses/schedules work based on a GPU memory, temperature and utilisation threshold and prevents simultaneous training/learning | 0 |

**At least 50% of scheduled model calls are reserved for verification.** The controller uses measured inference-call time as a second target, adding checks until verification occupies at least half the cycle or ten verifier calls are reached. This is a wall-time proxy, not a measurement of electrical energy or GPU compute. For model-weight promotion, a separate evaluator runs at least **60 paired tasks on each of two disjoint holdout sets** against both incumbent and candidate (at least 240 model generations). Promotion requires a positive main-set gain, no audit-set regression, and limited per-family regression. All labels are recomputed by deterministic maths code. This remains a deliberately narrow arithmetic proof-of-infrastructure.

The optimizer can improve prompts. The QLoRA trainer changes **candidate adapter weights**, never the base Ollama chat weights. Auto-promotion changes a versioned local manifest only after the verification gate; it does not upload or modify an Ollama model, alter guardrails, or grant arbitrary execution access.

## Windows 11: update an EXISTING local installation

**First:** close the current AI chat, learning and dashboard terminal windows before copying updated source files. The local SQLite database and `.venv` remain intact.

Open PowerShell **in your current old Continual AI project folder** (the folder with `START_ALL_WINDOWS.cmd`), then run:

```powershell
$target = (Get-Location).Path
$zip = Join-Path $env:TEMP 'continual-ai-v041.zip'
$tmp = Join-Path $env:TEMP 'continual-ai-update-v041'
Invoke-WebRequest 'https://github.com/RV-crtl/continual-ai-lab/archive/refs/heads/main.zip' -OutFile $zip
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
Expand-Archive $zip -DestinationPath $tmp
& (Join-Path $tmp 'continual-ai-lab-main\UPDATE_LOCAL_WINDOWS.ps1') -Target $target
```

The updater makes a `backups/source-TIMESTAMP` backup of source before replacing it, and runs tests. It does **not** overwrite `data/local`, `.venv`, `.venv-train`, chat histories, or local model files. If you have custom edits to source code, compare the backup before using the new code. To avoid copying files while they are in use, close the existing processes first.

Launch `START_ALL_WINDOWS.cmd`. It starts the multi-agent learner, read-only dashboard and regular Qwen3-4B chat. If `.venv-train` has been installed, it also launches the GPU training supervisor. You can always stop each terminal with Ctrl+C.

The separate `START_AGENTS_WINDOWS.cmd` launches the multi-agent loop without other windows.

## Enable real CUDA training on RTX 5070 Ti Laptop GPU (12 GB VRAM)

**Hardware:** 32 GB RAM and 12 GB NVIDIA VRAM. For native Windows QLoRA with Blackwell `sm_120`, use an up-to-date CUDA 12.8 capable PyTorch wheel and matching bitsandbytes. This installation is **optional** and requires downloads and disk space; start on **AC power**, keep ventilation clear. We cannot verify your actual driver/thermal behaviour remotely.

In PowerShell from the updated project folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL_TRAINING_WINDOWS.ps1
```

This builds `.venv-train` separately from your working chat environment, installs CUDA PyTorch, PEFT, Transformers and bitsandbytes, and checks whether GPU kernels actually execute. If the smoke test fails, **do not start training**: inspect the error and fix environment compatibility first. CPU learning and chat can continue.

Run `START_AUTO_TRAIN_WINDOWS.cmd` (or start `START_ALL_WINDOWS.cmd` once `.venv-train` is present). The training supervisor:

1. Waits for at least 96 unique, independently checked training labels and at least 96 new examples since its last training attempt.
2. Waits until NVIDIA reports >=7500 MiB free VRAM, temperature <79 °C and GPU utilisation <50%. Thresholds are conservative scheduling guards, **not** hardware safety guarantees.
3. Acquires `data/local/training.lock` so the multi-agent loop pauses its inference calls (existing chat is not forcibly stopped).
4. Supervises training and evaluation subprocesses with a wall-time limit and stops a process if GPU temperature remains >=84 °C for three ten-second checks. Loads `Qwen/Qwen3-1.7B` in 4-bit NF4 and trains an isolated LoRA adapter on up to 512 verified arithmetic samples, max 40 gradient update steps, batch size 1 and gradient accumulation 8.
5. Releases training memory by ending its subprocess; runs paired candidate-versus-incumbent tests on **two separate 60-task evaluation ranges**, never used as training examples.
6. Only if all gates pass, atomically writes `data/local/training/active.json` referencing the candidate adapter. Otherwise leaves the incumbent intact and records reasons in `data/local/training/reports`.
7. Repeats after 600 seconds; Ctrl+C stops. Candidate data/models, version pointers and reports stay private under `data/local`.

**Important:** The 4B Ollama chat is a different model and remains unchanged. To chat with a **promoted** 1.7B adapter, launch `START_TRAINED_CHAT_WINDOWS.cmd`. It will refuse to start until an adapter passes the gate. A promoted adapter is not automatically a better general assistant; it was evaluated only on the arithmetic curriculum. `python -m continual_ai.training_registry` is not a runnable CLI; rollback can be done via Python code or by restoring `active.json` from checkpoint/history.

### GPU contention and sustained operation

- Only one fine-tuning/evaluation subprocess uses the GPU at a time. The agent controller respects the training lock. The resource governor postpones training if inference or other processes are occupying GPU memory.
- Ollama has a 60-second keep-alive for this project's model requests to release idle VRAM sooner. Interactive chat activity can still occupy memory; training may defer until GPU resources become available.
- Do **not** expect fixed wattage or 50% physical energy consumption from the verification budget. Task runtimes are logged in `data/local/agent_cycles.jsonl` and available via the dashboard API.
- Low free disk space, battery operation, CUDA incompatibilities, excessive GPU temperature and high background utilisation are reasons to pause the trainer. Software sampling is a fallback, not a guarantee of safe operating temperatures. Periodically check NVIDIA temperatures and storage.

## Cloud version (GitHub Actions)

- Workflow: `.github/workflows/online-learning.yml`, or [Actions](https://github.com/RV-crtl/continual-ai-lab/actions).
- Scheduled **daily at 03:17 UTC** and on relevant source pushes; may be delayed or skipped by GitHub. Hosted CPU runners are ephemeral.
- Uses local Ollama `qwen3:1.7b`; runs **two four-call multi-agent cycles** per job and persists only public arithmetic attempts, labelled training examples, agent metrics and state to `data/cloud`.
- Hosted agents do **not** update neural weights. GitHub retains source code and measured results; local CUDA fine-tuning remains on your desktop. Never push local chats or checkpoints to this public repo.
- GITHUB_TOKEN contents write permission is scoped to the workflow; it will commit only `data/cloud` records. If a protected branch prevents the push, review GitHub Actions logs.

## Verification and auditing

```
python -m unittest discover -s tests -v
python -m compileall -q continual_ai
python -m continual_ai.multi_agent local --cycles 2
```

`tests/test_multiagent.py` checks the call allocation, independent verifier split, cloud resumption, rechecked training labels, and rejection gate. Tests do not need Ollama or CUDA. Live training and native-Windows installation **cannot be validated without running on your machine**; self-tests are not equivalent to a real neural training run.

### Security and data handling

The AI cannot modify its evaluator, change GPU governor settings, execute general shell commands, or push code autonomously. It can attempt bounded maths questions and select from fixed prompt strategies. Model downloads should come from official registry sources. Local chats and weights are excluded by `.gitignore`. GitHub Actions arithmetic logs are public. Do not publish `.env`, credentials, personal datasets, or `data/local` content.

## Learning limitations / next research steps

The 6-family arithmetic curriculum is small and can easily overfit. A promotion gate with 60 tasks per split cannot conclusively distinguish small statistical gains. For an advanced system, add independently sourced programming tests with isolated execution, research QA with cited evidence, distributed evaluation, calibration measurements, model quality drift checks, and broad held-out reasoning suites. Avoid automatic ingestion of unverified self-generated answers. This repository is a **practical, measurable learning infrastructure**, not the world's best AI.
