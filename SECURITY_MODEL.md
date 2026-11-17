# Gemma-Grook security model

Repository: `Ryukijano/Gemma-Grook` (research codebase: Gemma-Le / GR00T vision-language-action policy training and robot inference). No infrastructure repository exists: the Dockerfiles (`Dockerfile`, `orin.Dockerfile`) and SLURM scripts (`submit_job.sh`, `submit_notebook*.sh`, `scripts/*.sh`) are the whole deployment (user-confirmed).

## System overview and trust boundaries
- Training runs as SLURM batch jobs on a shared HPC GPU cluster (`submit_job.sh`, `scripts/gr00t_finetune.py`, `scripts/train_*.py`), pulling models/datasets from Hugging Face and logging to W&B, with `HF_TOKEN` in the job environment.
- Inference: a ZeroMQ REQ/REP server (`gr00t/eval/service.py`, `gr00t/eval/robot.py`, launched by `scripts/inference_service.py` / `scripts/simulation_service.py`) on port 5555, bound to `*` by default. Robot/sim clients connect over the network and receive actions that drive physical robots.
- **Deployment (user-confirmed):** the inference server runs on a **shared robot/lab LAN that other people use**. Other LAN users are untrusted relative to the server and robot.
- Trust boundaries: LAN client -> inference server; Hugging Face Hub / downloaded checkpoints -> training & inference processes; shared HPC cluster users -> job files, caches and env.

## Components
| Component | Code |
|---|---|
| Inference Server | `gr00t/eval/service.py`, `gr00t/eval/robot.py`, `gr00t/eval/simulation.py`, `scripts/inference_service.py`, `scripts/simulation_service.py` |
| Policy & Model Loading | `gr00t/model/**`, `gr00t/utils/**`, `code/lerobot/**/policies/gemma_le/**`, `deployment_scripts/**` |
| Training Pipeline | `gr00t/experiment/**`, `scripts/gr00t_finetune.py`, `scripts/train_*.py`, `scripts/eval_*.py`, `scripts/test_model*.py`, `gqe/**`, `qrisp_utils/**` |
| Model Publishing | `scripts/upload_to_hf.py`, `scripts/bulk_upload.py`, `scripts/export_weights.py` |
| Cluster & Container Infra | `Dockerfile`, `orin.Dockerfile`, `submit_*.sh`, `scripts/*.sh`, `accelerate_*.yaml`, `fsdp_config.json` |
| Vendored Third-Party | `code/lerobot/**` (except gemma_le), `gr00t/model/backbone/eagle2_hg_model/**`, `flash-attention/` |
| Examples & Notebooks | `getting_started/**`, `*.ipynb` |
| CI | `.github/**` |

## Entry points and who can reach them
- ZMQ inference endpoints `get_action`, `get_modality_config`, `ping`, `kill` — reachable by anyone on the shared lab LAN. Requests are deserialized with `torch.load(weights_only=False)` before the optional `api_token` check; `api_token` defaults to None (no auth).
- Model/checkpoint paths and HF repo ids passed on the CLI (`torch.load`, `snapshot_download`, `trust_remote_code=True`).
- SLURM job scripts and caches on shared `/scratch`.

## Assets (user: all roughly equal)
- Hugging Face token and private model weights (`Ryukijano/gemma-groot`, private by default).
- HPC/GPU cluster account and compute.
- Physical robot safety — actions from the server move real hardware.
- Datasets and W&B experiment data.

## Attacker profiles
- **Other users on the shared robot/lab LAN** (user-confirmed): can reach port 5555, send arbitrary bytes, and impersonate the server to clients.
- Other users of the shared HPC cluster (inferred: reads of world-readable files on `/scratch`, job logs).
- Supply chain: malicious HF model/dataset or dependency (inferred).

## Existing controls
- Optional shared `api_token` (plaintext, compared after deserialization).
- Upload scripts default to private HF repos; HF token read from env, not hardcoded.

## Out of scope / accepted risk (user-confirmed)
- Vendored third-party code: `code/lerobot` copy (except the project's own `gemma_le` policy), Eagle backbone (`gr00t/model/backbone/eagle2_hg_model`), `flash-attention`.
- CI workflows (`.github/**`).
- Notebooks and `getting_started` examples.

## Open questions
- Is `api_token` set in practice when serving on the lab LAN?
- Are committed W&B logs / `.err` files reviewed for secrets before pushing?
