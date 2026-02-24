# Volunteer Trial Project: Blood Cell Classification (BloodMNIST)

A collaborative trial project to build a multi-class blood cell classification pipeline using the [BloodMNIST](https://medmnist.com/) dataset. This trial assesses your ability to independently execute, collaborate, and adhere to project standards before onboarding to the main research project.

> [!IMPORTANT]
> **Read [COLLABORATORS.md](./COLLABORATORS.md) before writing any code.** It defines the conventions and standards you are expected to follow throughout this trial. Adherence to these standards is a core part of the evaluation.

---

## Goal

Build a working classification pipeline for BloodMNIST that includes:

1. **Data pipeline** — Download and prepare the dataset, generate and commit frozen train/val/test splits
2. **Model & training** — Implement a model, train it with full W&B logging (hyperparameters, training curves, final metrics, confusion matrix)
3. **Evaluation** — Report accuracy, weighted F1, per-class F1, and log a confusion matrix
4. **Best model** — Save the best model as a W&B Artifact (never committed to git)

Lighter weight models like ResNet-18 are perfectly fine for running locally, and ResNet-50 if your hardware allows. Whatever works best for your hardware is reasonable. 

How you structure the code, organize scripts, and divide work is up to you — the standards in [COLLABORATORS.md](./COLLABORATORS.md) are non-negotiable, but the implementation approach is yours to decide.

---

## Dataset

| Property | Detail |
|----------|--------|
| **Source** | [MedMNIST — BloodMNIST](https://medmnist.com/) via the `medmnist` package |
| **License** | CC BY 4.0 
| **Task** | Multi-class classification |

---

## W&B Tracking Requirements

Use a shared W&B project (e.g., `Volunteer-Trial-BloodMNIST`) so progress can be monitored.

| What to Log | How |
|-------------|-----|
| Hyperparameters | `wandb.config` (LR, batch size, epochs, model, etc.) |
| Training curves | Train/val loss and val accuracy per epoch |
| Final metrics | Accuracy, weighted F1, per-class F1 |
| Confusion matrix | `wandb.plot.confusion_matrix()` or equivalent |
| Best model | Save as a W&B Artifact (**never commit `.pth` to git**) |

---

## Hygiene

- **Isolate dependencies** in a `requirements.txt` so the environment is reproducible
- **Configure credentials** (e.g., `WANDB_API_KEY`) via a `.env` file that is git-ignored — see [COLLABORATORS.md §4](./COLLABORATORS.md) for details
