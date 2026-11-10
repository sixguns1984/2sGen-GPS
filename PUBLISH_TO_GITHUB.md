# Publishing this repository to GitHub

This folder is already structured as the GitHub repository root. Do not place the entire folder inside an extra nested subdirectory.

## Upload through Git (recommended)

```bash
# After extracting the release ZIP:
cd 2sGen-GPS_GitHub_release
git init
git branch -M main
git remote add origin https://github.com/sixguns1984/2sGen-GPS.git
git add .
git commit -m "Release CTAR-MPTGA pip package and manuscript analysis pipeline"
git push -u origin main
```

If the GitHub repository contains an existing main branch/README or commit history, do not force-push. Instead clone it, copy the contents of this package into the clone, then `git add -A`, commit and push.

## Install after upload

```bash
pip install 'git+https://github.com/sixguns1984/2sGen-GPS.git'
2sgengps --help
```

To support the bare `pip install 2sGen-GPS` command, a separate PyPI release and account authorization is required. GitHub hosting alone is not PyPI publication.

## Before journal submission

- MIT License is included; confirm that the included source code may be redistributed under MIT.
- Run complete controlled-data pipeline and check the 22,842-pair results against final manuscript.
- Confirm final LD cutoff from original logs.
- Resolve cross-lag SC comparability if relevant to final reported selections.
- Confirm postmortem five-region QC and pseudobulk provenance.
- Remove local server paths or replace with configurable parameters where feasible.
- Tag an immutable GitHub release and optionally archive it in Zenodo.
