# overleaf/

Manuscript source for the Russian River seepage paper.

This folder **is** the Overleaf project: it is a git clone connected directly to
Overleaf (`origin = https://git.overleaf.com/65819c319d0b4dfe15326e00`). Edit the
files here and push/pull directly — there is no separate `draft/`/`mirror/` staging.

## Workflow

```bash
cd overleaf

# Get the latest changes from Overleaf before editing
git pull

# ... edit main.tex, local.bib, figures/, etc. ...

# Send your changes to Overleaf
git add -A
git commit -m "Describe your changes"
git push
```

If `git push` is rejected because Overleaf has newer commits, run `git pull` first
(resolve any conflicts), then push again.

## Notes

- `snapshot_*/` folders are frozen, local-only snapshots — do not edit, and they
  are git-ignored so they are never pushed to Overleaf.
- `README.md`, `.DS_Store`, and LaTeX build artifacts are git-ignored and stay local.
- Before major revisions, create a new `snapshot_<date>/` folder from the current files.
