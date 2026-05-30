# Contributing

## Local checks

Before opening a PR, run the same checks CI runs:

```bash
python -m compileall -q src main.py        # byte-compile
python scripts/ci_smoke.py                 # 1-step PPO training smoke (CPU)
```

CI (`.github/workflows/ci.yml`) runs these on every push/PR, and CodeQL
(`.github/workflows/codeql.yml`) runs a security scan.

## Commit messages

Conventional Commit style is encouraged for readable history:

```
feat: add belief auxiliary head
fix: correct GAE reset across auto-reset episodes
docs: clarify gradient routing
```

(This repo does not auto-release; commit style is for clarity only.)
