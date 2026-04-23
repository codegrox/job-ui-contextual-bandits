# Job UI Contextual Bandits

Large-scale contextual bandit simulation for adaptive job-application UI selection.

This project models a job platform that chooses one of five UI layouts for each user session and job opportunity, with the goal of improving early-stage application outcomes while reducing wasted applications.

## Project summary

The simulator evaluates five interface arms:
- `panel_split` - LinkedIn-style split pane with job details visible next to the list
- `swipe_fast` - fast swipe flow optimized for rapid decisions
- `card_grid` - dense grid layout for broad exploration
- `guided_chat` - conversational guided interface
- `hybrid_ranked` - ranked hybrid interface balancing preview and speed

At each interaction, the platform observes a context vector built from:
- applicant attributes
- company attributes
- job attributes
- session intent / behavior signals

The benchmark compares:
- `epsilon_greedy`
- `ucb1`
- `gaussian_ts`
- `linucb`
- `contextual_ts`

## Main result

Across the final experiment bundle, the contextual methods are strongest. In the public-facing report figure pipeline, **LinUCB** achieves the lowest cumulative regret and the highest offline replay reward, with **Contextual Thompson Sampling** as the closest competitor.

The environment is genuinely contextual: different archetypes and session goals favor different UI layouts, so routing everything to one global interface leaves performance on the table.

## Repository structure

```text
config.py                      project configuration
run_all.bat / run_all.sh       convenience runners
requirements.txt               Python dependencies
src/                           main pipeline code
notebooks/                     keep only one final notebook here
outputs/figures/               exported plots
outputs/tables/                summary tables and selected result artifacts
report/                        final LaTeX report source / PDF
```

## One notebook to rule the repo

For the final public repo, keep a **single main notebook**:

```text
notebooks/project_report_pipeline.ipynb
```

This notebook:
- generates the polished report figures
- exports the summary CSV used in the report
- serves as the easiest entry point for graders / recruiters / visitors

If you are cleaning the repo, the older exploratory notebooks can be removed after this final notebook is in place.

## Environment setup

### Windows CMD

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install ipykernel jupyter
python -m ipykernel install --user --name bt4014-env --display-name "Python (bt4014-env)"
```

### Linux / macOS shell

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install ipykernel jupyter
python -m ipykernel install --user --name bt4014-env --display-name "Python (bt4014-env)"
```

## Required tools

- Python 3.11
- Git
- Git LFS
- Jupyter

## Running the full data pipeline

### Debug run

```cmd
run_all.bat debug
```

### Medium run

```cmd
run_all.bat medium
```

### Full run

```cmd
run_all.bat full
```

## Reproducing the final report figures

Open Jupyter and run:

```cmd
jupyter lab
```

Then open:

```text
notebooks/project_report_pipeline.ipynb
```

Run all cells from top to bottom.

The notebook exports these main artifacts into `outputs/figures/`:
- `01_simulator_world_stats.png`
- `02_reward_landscape_full.png`
- `03_bandit_dashboard.png`
- `04_offline_bootstrap_replay.png`
- `05_behavioural_signals.png`
- `06_hyperparameter_and_ablation.png`
- `07_session_goal_routing.png`
- `policy_summary_table.csv`

## Large generated artifacts

This repository can include large generated results tracked with **Git LFS**.

If you clone the repo, install Git LFS before pulling:

```cmd
git lfs install
git lfs pull
```

If the data is too large to upload to Canvas directly, the BT4014 project guidelines explicitly allow providing a GitHub link for the code/data bundle.

## License

This repository is licensed under the MIT License. See `LICENSE` for details.

### Third-party and non-project materials

Some files used during development may originate from course resources, third-party templates, or external references. Those materials are **not automatically covered by the MIT License** for the repository and remain subject to their original copyright or license terms.
