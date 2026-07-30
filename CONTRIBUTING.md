[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
# Contributing to AnomalyMatch

Thank you for your interest in improving AnomalyMatch! AnomalyMatch is developed
by the ESAC Data Science team at the European Space Agency (ESA) and released
under the ESA Public License (ESA-PL).

## Read this first: community governance

The rules that apply to all our projects (code of conduct, contribution
workflow, Contributor License Agreement, and licensing) live in one place:

> **[ESAC Data Science Community Governance](https://www.cosmos.esa.int/web/data-science/contributing-to-our-software)**

**Please read it before opening a pull request.** In particular:

- **A signed Contributor License Agreement (CLA) is required before we can merge
  your contribution.** Signing it confirms that the work is yours to give and
  grants ESA the right to redistribute it under the project license. You keep
  the copyright to your contribution, and you only need to sign once. The form,
  the submission address, and the full explanation are in the governance
  document.
- For **small changes** (typos, small fixes, documentation), open a pull request
  directly. For **larger changes**, open an issue first so we can scope the work
  together.
- Please report security vulnerabilities privately, not in a public issue.

The rest of this document covers only the technical specifics of contributing to
AnomalyMatch.

## Issues and bug reports

Open issues via the GitHub issue tracker and use the provided templates. Clear
reproduction steps, your environment details (OS, Python version, GPU and CUDA
version, AnomalyMatch version), and the expected versus actual behaviour help us
resolve things quickly.

## Development environment

Dependencies are declared in `environment.yml` and installed with conda:

```bash
conda env create -f environment.yml
conda activate am
pip install -e .          # editable install for development
```

A GPU is strongly recommended for anything involving training or prediction over
large datasets. The UI relies on ipywidgets, so a Jupyter environment is the
easiest way to exercise it (see `StarterNotebook.ipynb`).

## Branching model

AnomalyMatch uses a two-branch model:

- **`develop`** is the integration branch. **Target `develop` for features,
  fixes, and refactors.**
- **`main`** is reserved for releases and hotfixes.

Stacking a pull request on another feature branch is fine; otherwise branch from
and target `develop`. Use descriptive branch names with the same prefix as your
commit type (`feat/...`, `fix/...`, `docs/...`).

## Coding conventions

- **Commits and branches:** follow
  [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`) and
  use matching branch prefixes.
- **Formatting and linting:** code is linted and formatted with
  [Ruff](https://docs.astral.sh/ruff/) (line length 100). Run both before
  pushing:

  ```bash
  ruff check .
  ruff format .
  ```

- **License headers:** every source file must start with the standard ESA license
  header. Copy it from any existing file of the same type. CI rejects files
  without it.
- **Style:** follow PEP 8 and document concisely. Comments should explain why,
  not restate what the code does.
- **Configuration:** configuration is accessed through DotMap with direct field
  access (`cfg.a.b`). Do not use `getattr()` or `.get()` fallbacks for config
  values. Defaults live in `anomaly_match/utils/get_default_cfg.py`.

## Tests

Add or update tests for your change, and make sure the suite passes locally
before pushing:

```bash
pytest                              # full suite
pytest --cov=anomaly_match tests/   # with coverage
pytest tests/unit/test_file.py -v   # a single file
```

## Continuous integration

Only pull requests that pass all CI checks are merged. CI runs:

- License-header check
- Ruff lint and format check
- Dead-code (Vulture) check
- Test suite

Please make sure these pass locally first. CI is a safety net, not a substitute
for running the checks yourself.

## Questions

Not sure about something? Open an issue. We are happy to help.
