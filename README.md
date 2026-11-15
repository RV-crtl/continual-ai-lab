# Continual AI Lab — Local + GitHub learning prototype

A **working research starter** for two independent AI experiments:

1. **Local Windows AI**: native Windows Ollama chat, persistent chat history, continuously repeated verifiable tasks, SQLite-based verified experiences, local statistics and an optional **real LoRA weight-training** pipeline.
2. **GitHub-hosted online AI**: a scheduled GitHub Actions workflow running a small Ollama model on an ephemeral CPU machine, persisting verified task attempts and labelled training examples in Git history.

**This is not an AGI system, and improvement is not guaranteed.** The initial learning curriculum covers only deterministic arithmetic and small algebra problems. Stored experiences, few-shot prompts and fine-tuned adapters are different improvement mechanisms. **The GitHub version does not change the model's weights.** The local LoRA trainer saves a *candidate*, not an automatically deployed model. Nothing uses unverified AI-generated facts as training labels.

## Computer requirements

The provided hardware report confirms Windows 11, Intel Core Ultra 9 275HX, 32 GB system RAM, NVIDIA RTX 5070 Ti Laptop GPU with **12 GB VRAM** (~8.5 GB free at measurement), and approximately **157 GiB free disk**. Start with `qwen3:4b` (approximately 2.5 GB download in Ollama). A dedicated NVIDIA GPU with sufficient VRAM can make model inference and training substantially faster. Close memory-intensive programs before starting if RAM is nearly full. The chosen 4B model is deliberately conservative. A quantised 8B model may fit, depending on context size and other GPU allocations. Real 4B LoRA/QLoRA training requires a separate CUDA-capable training setup and is NOT the default training recipe in this release.

- Windows 11; Python 3.11–3.13; current Ollama Windows installation.
- Ollama download: <https://ollama.com/download/windows>
- Python: <https://www.python.org/downloads/>
- No Docker, WSL or VM is required for initial native Windows chat, learn, benchmark and dashboard.
- Real LoRA training additionally needs a suitable PyTorch installation and preferably a dedicated NVIDIA GPU. Consult <https://unsloth.ai/docs/get-started/install/windows-installation> for up-to-date native Windows training options.

## Windows quick start (one folder, three launchers)

**Step 1.** Extract this ZIP to a persistent location, e.g. `C:\Continual-AI-Starter`. Install Python and Ollama if not already installed. Ollama generally starts in the background after Windows installation.

**Step 2.** Open PowerShell in that directory and execute:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL_WINDOWS.ps1
```

The installer creates `.venv`, runs the unit tests, and downloads `qwen3:4b` if Ollama is available. It does not install system-wide Python or Ollama without your involvement. Alternatively, install Ollama manually and use:

```powershell
ollama pull qwen3:4b
.\.venv\Scripts\python.exe -m continual_ai.cli doctor
```

**Step 3.** Double-click `START_ALL_WINDOWS.cmd` to open chat, learning and dashboard together. Or launch these components separately:

- `START_CHAT_WINDOWS.cmd` — chat with the local model.
- `START_LEARNING_WINDOWS.cmd` — run the ongoing independently verified arithmetic/algebra curriculum until you press **Ctrl+C**.
- `START_DASHBOARD_WINDOWS.cmd` — view local stats at <http://127.0.0.1:8765>.

You can use terminal commands instead:

```powershell
.\.venv\Scripts\python.exe -m continual_ai.cli doctor
.\.venv\Scripts\python.exe -m continual_ai.cli chat
.\.venv\Scripts\python.exe -m continual_ai.cli learn --rounds 12
.\.venv\Scripts\python.exe -m continual_ai.cli loop --pause 2
.\.venv\Scripts\python.exe -m continual_ai.cli benchmark --n 24
.\.venv\Scripts\python.exe -m continual_ai.cli export
```

**To select a different model**, use `ollama pull qwen3:8b` and append `--model qwen3:8b` to a chat, learn or loop command. The default is deliberately conservative because GPU VRAM is unknown.

### Find the correct model for the actual target desktop

Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\CHECK_HARDWARE_WINDOWS.ps1` **on the desktop that will run the model**. The available report describes the Lenovo Legion laptop. If using a different desktop, run the diagnostic on THAT computer to choose its model. The report shows the GPU name, NVIDIA VRAM (when applicable), CPU and free storage. Share that output before choosing a larger model or attempting GPU-intensive fine-tuning.

### What happens on each learning task?

1. A deterministic task generator creates a fresh maths problem and independently computes its correct answer.
2. The model attempts the problem, optionally receiving up to two previously verified examples of the same task family.
3. A strict answer parser independently compares the model's response against the computed answer.
4. The complete attempt and verified mathematical label are saved in local SQLite.
5. Future runs retrieve verified few-shot examples and accumulate a supervised-training dataset.
6. A separate held-out benchmark uses seeds outside the learning range and is never put into the training set.

This establishes **verified-experience learning**, not guaranteed gain in model accuracy. The model's internal weights remain unchanged during `loop`.

### Persistent state and privacy

- `data/local/learning.sqlite3`: locally stored task attempts and chats. Excluded from Git.
- `data/local/verified_sft.jsonl`: independently labelled local supervised data. Excluded from Git.
- `data/local/benchmarks.jsonl`: local held-out scores. Excluded from Git.
- `data/cloud/`: hosted experiments only. These records are **committed to GitHub**; no local chat is uploaded.
- Dashboard binds to `127.0.0.1`, not the public network.
- There is **no general shell/code execution** by AI-generated instructions. The model cannot rewrite its own safety checks or evaluator.

## Optional real LoRA model training (local only)

**Warning:** Training can exhaust VRAM/RAM and may be slow or impractical without an NVIDIA GPU. Train in an isolated environment, keep the laptop plugged in, and monitor temperatures. The starter uses a small model as the training target because your GPU hasn't been established. The first training run is a proof of functionality, not a claim of better general intelligence.

After collecting at least 32 unique verified task labels (hundreds are better), create a *separate* training environment with GPU-enabled PyTorch, install the optional requirements and run:

```powershell
.\.venv\Scripts\python.exe -m continual_ai.cli export
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -m continual_ai.train_lora --dataset data/local/verified_sft.jsonl --base-model Qwen/Qwen3-0.6B --output-dir data/local/candidates/adapter-v1
.\.venv\Scripts\python.exe -m continual_ai.benchmark_adapter --base-model Qwen/Qwen3-0.6B --adapter data/local/candidates/adapter-v1 --n 30
```

`train_lora` actually updates the LoRA weights on the Hugging Face `Qwen/Qwen3-0.6B` base. `benchmark_adapter` measures the pretrained base and candidate on separately seeded tasks and emits a JSON verdict. **No model is automatically promoted or loaded in Ollama**. Ollama `qwen3:4b` and this 0.6B training branch are different base models: adapters cannot be transferred between them. For production, use a model-matched trainer, larger benchmarks, test isolation, and a reviewed deployment gate.

If training libraries conflict on native Windows, do not install random packages blindly: isolate training in Conda or WSL2 according to your GPU and current official documentation. Fine-tuning 4B/8B/27B needs a separate hardware assessment.

## GitHub-hosted version — scheduled, not 24/7

GitHub Actions will run the small Ollama `qwen3:0.6b` model daily, attempt eight independently verifiable problems, store each attempt, and commit its verified examples. A static summary is written to `data/cloud/summary.md`. The small model is chosen for normal CPU runners, not for top-tier performance. A source-code push or manual dispatch can trigger an initial run, in addition to the scheduled run.

**IMPORTANT:** `INSTALL_WINDOWS.ps1` saying **Setup complete** refers to local Python/Ollama only. It does **not** create a GitHub repository. The next publish step is independent.

**How to publish to GitHub** (the current `RV-crtl/continual-ai-lab` repository is PUBLIC):

1. Install [Git](https://git-scm.com/download/win) and [GitHub CLI](https://cli.github.com/).
2. Open a new PowerShell in the extracted folder and sign in using `gh auth login`.
3. Run `PUBLISH_GITHUB_WINDOWS.cmd` by double-clicking it **or** execute `powershell -NoProfile -ExecutionPolicy Bypass -File .\PUBLISH_GITHUB_WINDOWS.ps1`. The script checks that GitHub CLI is signed in to **RV-crtl**, then updates **RV-crtl/continual-ai-lab** (or can create it as a private repository if it does not already exist). It prints the exact URL only after remote verification.
4. On GitHub, open **Actions** and manually run the workflow named `Online learning experiment (CPU, verified memory)` once to verify the environment and permissions; alternatively run `gh workflow run online-learning.yml --repo RV-crtl/continual-ai-lab`. This may consume metered GitHub Actions minutes.
5. Confirm `data/cloud/summary.md` and versioned JSONL files appear after the run. Then the cron schedule will execute on the default branch when GitHub schedules it. The linked ChatGPT GitHub App cannot create a repository; the GitHub CLI is required for this one-time step. The app installation must be able to access new repositories to manage them in later conversations.

The publisher targets **only** `RV-crtl/continual-ai-lab`, verifies the authenticated account and uses a non-force push. The current repository is public: never publish private prompts, personal records, access credentials, model checkpoints or local chat logs. GitHub repository publication does not mean a hosted model is continuously running.

A GitHub Actions schedule may be delayed or skipped during platform load. Hosted runners are temporary, usually without useful training GPUs, and private repository minutes are metered. GitHub's default `GITHUB_TOKEN` must permit `contents: write` (subject to repository and organisation settings). The workflow limits each job to 35 minutes and records state only after verified tasks. Repository history grows with each run. Standard scheduled runs are NOT a permanently operating AI service.

GitHub docs: <https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows> and <https://docs.github.com/en/actions/reference/runners/github-hosted-runners>.

## Model improvement limitations (important)

| Capability | Local | GitHub Actions |
| --- | --- | --- |
| Chat with a downloadable LLM | Yes | Batch inference only |
| Retain independently verified memories | Yes | Yes, committed JSONL |
| Memory-assisted task attempts | Yes | Yes |
| Keep working after the terminal closes | No; needs a persistent process or OS task | Scheduled independently |
| Change neural network parameters | Optional LoRA candidate training | No |
| Automatically promote a weight update | No, deliberately gated | No |
| Validate broad general intelligence | No; maths smoke test only | No |

### Engineering roadmap to a more capable system

- Introduce external verified programming tests (locked-down sandbox), proof checking and research retrieval with source provenance.
- Expand the benchmark into a sufficiently sized, frozen, leakage-resistant suite across coding, reasoning, instruction following and safety.
- Train a candidate adapter periodically, evaluate capability regressions and latency, and promote only after passing statistical and safety gates.
- For always-on remote availability: deploy the **same source** on a persistent GPU server or your own self-hosted runner; keep GitHub as source control and CI, not as the 24/7 model host.
- Add user approvals for proposed self-modifications, signed releases, resource budgets, circuit breakers and a rollback mechanism before allowing autonomous code edits.

## Troubleshooting

- **Can't connect to Ollama**: start Ollama from the Windows Start menu; check `http://127.0.0.1:11434/api/tags` on your machine.
- **Not enough RAM**: close programs, run `ollama pull qwen3:1.7b`, then add `--model qwen3:1.7b` to the CLI command. Model weight files are not equivalent to peak process memory usage.
- **Ollama says model unavailable**: run `ollama list` and pull the tag you intend to use.
- **Slow learning**: CPU inference on a small model can still be slow. Each task waits for inference to finish. Reduce the model size if needed.
- **GitHub workflow permission error**: check repository Actions workflow permissions; it needs `contents: write` solely to push `data/cloud/*` after the job.
- **GitHub Actions disabled or suspended**: enable Actions for the new repo and inspect account quota, permissions, job logs and platform rules.
- **Model writes incorrect answer formatting**: the verifier will mark it as a failure; this is intentional and does not produce untrusted success labels.

## Safety and quality

- No arbitrary AI-generated code is run. All task ground truths are independently calculated.
- No auto-push of modified source code. Only dedicated online verified-data paths are automatically committed.
- GitHub Actions auto-installs Ollama through its current official install script; for a security-hardened deployment pin and review the installer and third-party actions.
- Never put private chats, access tokens, proprietary documents or personal files under `data/cloud/`.
- Stop local `loop` with Ctrl+C; saved records survive. Stop online runs via the GitHub Actions UI and disable the scheduled workflow to prevent future runs.

License: No license file included. Choose a repository license after reviewing the licences of any third-party models and tools you redistribute.
