# New Collaborator Reference

Read this before making changes. Ordered by importance.

---

## 1. Reproducibility

**Goal**: Given the same code, data, and hardware, anyone should get identical results.

This goes beyond just setting a random seed. Key areas to address:

- **Seeding**: Python, NumPy, PyTorch (CPU + CUDA)
- **Deterministic backends**: cuDNN, CUBLAS workspace config
- **DataLoader workers**: Seeded generators and worker init functions
- **Iteration order**: Use `sorted()` whenever iterating over files, directories, or any structure where order could vary across platforms (e.g., `os.listdir()`)
- **Dependencies**: Track which package versions you used (see `requirements.txt`) so the environment can be recreated reliably

Include a `reproducibility.py` module and wire it into all train/eval entry points. Set seeds consistently and enforce stable behavior (deterministic settings where applicable, stable iteration order, reproducible split usage). The same setup + same seed should produce consistent outcomes.

Read the [official PyTorch reproducibility docs](https://pytorch.org/docs/stable/notes/randomness.html) and set up reproducibility properly for your specific experiment.

> [!CAUTION]
> AI can help with reproducibility code, but do NOT blindly trust its output. It will hallucinate settings and silently break experiment validity. **Always validate against the [official PyTorch docs](https://pytorch.org/docs/stable/notes/randomness.html).**

---

## 2. Licensing & Attribution

**Any third-party dataset, code, or pretrained model you introduce must be properly attributed.**

- Create a `THIRD_PARTY_LICENSES.md` file in the repo for all attribution entries
- BloodMNIST usage must include a clear **CC BY 4.0** mention with a source link
- For each entry, include relevant details (e.g., source, license type, link) and read the actual license page for full coverage of license terms

> [!WARNING]
> Omitting attribution or ignoring license terms (e.g., NonCommercial, ShareAlike) can invalidate the project's ability to publish or share results. When in doubt, check the license deed before proceeding.

> [!CAUTION]
> AI can help draft licensing and attribution text, but do NOT blindly trust its output. It can misinterpret license terms, omit required clauses, or fabricate compatible licenses. **Always validate against the actual [license deed](https://creativecommons.org/licenses/) yourself.**

---

## 3. Never Hardcode Paths or Hyperparameters — Use `config.py`

All paths, defaults, W&B settings, and training hyperparameters live in `src/config.py`. All scripts import from it and accept CLI args for overrides. This means any collaborator can run the same code without hunting through files to change values.

---

## 4. Never Hardcode Secrets

API keys (e.g., `WANDB_API_KEY`) go in `.env`, which is git-ignored. **Never hardcode secrets in source files.** A leaked key pollutes git history permanently, creates security risk, and slows everyone down with key rotation.

---

## 5. What NOT to Commit

| Pattern | Why |
|---------|-----|
| `*.pth` | Model weights are large — save to W&B |
| `data/` | Downloaded via `dataset-setup/` scripts |
| `wandb/` | Local logs — already synced to cloud |
| `.env` | Secrets |
| `**/smoke-test/` | Always recreatable |

Models and logs are tracked via W&B artifacts, not git.

---

## 6. Dataset Splits are Frozen

`dataset_splits.json` defines which samples belong to the train, validation, and test sets. It is **git-versioned** so every collaborator trains and evaluates on identical partitions. Generate it once, commit it, and do not regenerate it during the trial.

---

## 7. Run from the Repo Root

All commands use `python -m` module syntax. Never `cd` into subdirectories.

```bash
# ✅ Correct
python -m experiments.bloodmnist.src.train

# ❌ Breaks imports
cd experiments/bloodmnist && python src/train.py
```

---

## 8. Communication

- Create a group chat (e.g., Discord, Slack, or similar) to coordinate work among yourselves.
- The maintainer will join the group chat to stay in the loop.
- The maintainer is available for clarifying questions (e.g., interpreting requirements, understanding conventions) but will not assist with implementation. The trial is meant to assess independent execution.

---

## 9. AI Use Policy

AI tools (e.g., Copilot, ChatGPT, etc.) are permitted. That said, AI-generated code is held to the same bar as hand-written code — adherence to the standards and practices outlined in this document is what determines an effective trial project, not how the code was produced.
