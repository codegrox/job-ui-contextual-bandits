# Job UI Contextual Bandits

Large-scale contextual bandit simulation for adaptive job-application UI selection.

This project models a job platform that chooses one of five UI layouts for each user session and job opportunity, with the goal of improving early-stage application outcomes while reducing wasted applications.

## Project summary

The simulator evaluates five interface arms:
- `panel_split` — LinkedIn-style split pane with job details visible next to the list
- `swipe_fast` — fast swipe flow optimized for rapid decisions
- `card_grid` — dense grid layout for broad exploration
- `guided_chat` — conversational guided interface
- `hybrid_ranked` — ranked hybrid interface balancing preview and speed

At each interaction, the platform observes a context vector built from:
- applicant attributes
- company attributes
- job attributes
- session intent / behavior signals

The platform chooses one UI arm and receives a reward based on first-stage outcomes:
- `interview_1`
- `rejected`
- `ignored`

The benchmark compares:
- `epsilon_greedy`
- `ucb1`
- `gaussian_ts`
- `linucb`
- `contextual_ts`

## Main result

The full benchmark covers **1,050,000 interactions**. The strongest method is **Contextual Thompson Sampling**, with **LinUCB** as the closest competitor. Both contextual methods outperform the non-contextual baselines on the main metrics.

The environment is genuinely contextual: different session goals favor different UI layouts.

## Repository structure

```text
config.py                      project configuration
run_all.bat / run_all.sh       convenience runners
requirements.txt               Python dependencies
src/                           main pipeline code
notebooks/                     plotting and export notebooks
data/raw/                      raw generated entities and arm profiles
data/summaries/                lightweight and medium-weight diagnostics
outputs/figures/               exported plots
outputs/tables/                summary tables and selected result artifacts
```

## Included artifacts

This public repository keeps:
- source code
- notebooks
- exported figures
- lightweight summary tables
- selected generated artifacts tracked with **Git LFS**

If you clone this repository, make sure **Git LFS is installed** before pulling the large files.

## Environment setup

### Windows CMD

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS shell

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Required tools

- Python 3.11
- Git
- Git LFS
- Jupyter Notebook

## Running the pipeline

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

## Reproducing generated artifacts locally

The large generated files in this repo can also be regenerated from scratch.

### 1. Generate the full simulated dataset and processed tables

```cmd
run_all.bat full
```

This recreates the main generated artifacts, including:
- `data/logged/rounds_chunk_*.parquet`
- `data/processed/oracle_summary.parquet`
- `data/processed/rounds_all_meta.parquet`
- `data/processed/train_rounds.parquet`
- `data/processed/valid_rounds.parquet`
- `data/processed/test_rounds.parquet`
- `outputs/tables/experiment_results_full_online_none.parquet`

### 2. Export report tables

```cmd
python -m src.export_report_tables --preset full
```

### 3. Export figures

```cmd
jupyter notebook
```

Then run:
- `notebooks/02_reward_landscape.ipynb`
- `notebooks/03_bandit_results.ipynb`
- `notebooks/04_report_figures.ipynb`
- optionally `notebooks/05_final_additional_figures.ipynb`

## Main figures

Core figures:
- `outputs/figures/reward_landscape_heatmap.png`
- `outputs/figures/cumulative_contextual_regret.png`
- `outputs/figures/arm_selection_frequency.png`
- `outputs/tables/algo_summary.csv`

Backup figures:
- `outputs/figures/rolling_avg_reward.png`
- `outputs/figures/outcome_mix_by_algorithm.png`

## Notes

- Heatmap stars are **row-wise**: one star marks the best arm within each session-goal row.
- If you clone without Git LFS, large tracked artifacts will download as small pointer files instead of the actual data.
- The full experiment and data-generation pipeline is computationally heavy; the included generated artifacts are provided to save rerun time.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for details.

### Third-party and non-project materials
Some files used during development may originate from third parties, course resources, publishers, or template distributions. Those materials are **not necessarily covered by the MIT License** for this repository and remain subject to their original licenses, copyright terms, or usage restrictions.

If you reuse this repository publicly, review and remove any third-party files that you do not have permission to redistribute.
