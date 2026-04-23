from pathlib import Path
import zipfile


def add_dir(zf, folder, root):
    for path in folder.rglob('*'):
        if path.is_file():
            arc = path.relative_to(root)
            zf.write(path, arc.as_posix())


def main():
    cwd = Path.cwd().resolve()
    root = cwd.parent if cwd.name == 'src' else cwd
    out = root / 'submission' / 'Codes_FINAL.zip'
    out.parent.mkdir(parents=True, exist_ok=True)

    include_paths = [
        root / 'src',
        root / 'notebooks',
        root / 'submission',
        root / 'outputs' / 'figures',
        root / 'outputs' / 'tables',
    ]
    include_files = [
        root / 'README.md',
        root / 'requirements.txt',
        root / 'config.py',
        root / 'run_all.bat',
        root / 'run_all.sh',
    ]

    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in include_files:
            if f.exists():
                zf.write(f, f.relative_to(root).as_posix())
        for d in include_paths:
            if d.exists():
                add_dir(zf, d, root)

    print('Wrote', out)


if __name__ == '__main__':
    main()
