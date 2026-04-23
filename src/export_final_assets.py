from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt


def main():
    cwd = Path.cwd().resolve()
    root = cwd.parent if cwd.name == 'src' else cwd
    fig_dir = root / 'outputs' / 'figures'
    table_dir = root / 'outputs' / 'tables'
    pres_dir = root / 'outputs' / 'presentation_data'
    pres_dir.mkdir(parents=True, exist_ok=True)

    # Copy lightweight CSVs used by teammates / Excel work
    for name in ['algo_summary.csv', 'arm_selection_frequency.csv', 'world_summary.csv']:
        p = table_dir / name
        if p.exists():
            shutil.copy2(p, pres_dir / name)

    # Export a compact runtime summary if logs exist
    logs_dir = root / 'outputs' / 'logs'
    runtime_rows = []
    for logname in ['generation.log', 'diagnostics.log', 'experiments.log']:
        p = logs_dir / logname
        if not p.exists():
            continue
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [x.strip() for x in line.split(',')]
                row = {}
                for part in parts:
                    if '=' in part:
                        k, v = part.split('=', 1)
                        row[k] = v
                if row:
                    runtime_rows.append(row)
    if runtime_rows:
        pd.DataFrame(runtime_rows).to_csv(pres_dir / 'runtime_summary.csv', index=False)

    # Simple runtime breakdown plot if runtime_summary exists
    rt_path = pres_dir / 'runtime_summary.csv'
    if rt_path.exists():
        rt = pd.read_csv(rt_path)
        if 'seconds' in rt.columns and 'preset' in rt.columns:
            rt['seconds'] = pd.to_numeric(rt['seconds'], errors='coerce')
            rt = rt.dropna(subset=['seconds'])
            if len(rt):
                fig, ax = plt.subplots(figsize=(10, 5))
                labels = [f"{r.get('preset','?')}:{r.get('chunk', r.get('rows',''))}" for _, r in rt.iterrows()]
                ax.bar(range(len(rt)), rt['seconds'])
                ax.set_title('Runtime breakdown')
                ax.set_ylabel('Seconds')
                ax.set_xticks(range(len(rt)))
                ax.set_xticklabels(labels, rotation=90)
                fig.tight_layout()
                fig.savefig(fig_dir / 'runtime_breakdown.png', dpi=220, bbox_inches='tight')
                plt.close(fig)

    print('Exported final lightweight assets to', pres_dir)


if __name__ == '__main__':
    main()
