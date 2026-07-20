[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)
# Contributing to AnomalyMatch

Thank you for your interest in improving AnomalyMatch! AnomalyMatch is developed
by the European Space Agency (ESA) and released under the ESA Public License
(ESA-PL). These guidelines keep contributions smooth for everyone. This process
is still evolving, so feedback on the guidelines themselves is welcome too.

## Issues and bug reports

We welcome all issues, bug reports, and feature requests — please open one via
the GitHub issue tracker and use the provided templates where available. Clear
reproduction steps, your environment details, and expected vs. actual behaviour
help us resolve things quickly.

## Contributor License Agreement (required before merge)

Because contributions are redistributed under an ESA open-source license, we can
only merge your changes once we have a signed Contributor License Agreement (CLA)
on file. The CLA clarifies the intellectual-property terms and protects both you
and the project.

1. Download the CLA from **[TBD — link to CLA download page]**.
2. Fill it in and sign it.
3. Send the signed agreement to **[TBD — submission email address]**, with
   **pablo.gomez at esa.int** in CC.

Due to intellectual-property requirements, **we can only merge your contribution
once we have received your signed CLA.** You only need to sign it once — it
covers your future contributions.

## Pull requests

Contributions are accepted as pull requests against the `main` branch.

- For small changes (typos, small bug fixes, docs), feel free to open a pull
  request directly.
- For larger or non-trivial changes, please **open an issue first** so we can
  scope the work together and make sure it fits the project's direction before
  you invest significant effort.
- Keep each pull request focused on a single logical change — it is much easier
  to review.

## Coding conventions

- **Commits & branches:** Follow [Conventional Commits](https://www.conventionalcommits.org/)
  for commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`,
  `ci:`, …) and use matching branch prefixes (`feat/…`, `fix/…`, `docs/…`).
- **Formatting & linting:** Code is linted and formatted with
  [Ruff](https://docs.astral.sh/ruff/). Run `ruff check .` and `ruff format .`
  before pushing.
- **License headers:** Every source file must start with the standard ESA license
  header — copy it from any existing file of the same type. CI rejects files
  without it.
- **Tests:** Add or update tests for your change and make sure the suite passes
  locally (`pytest --cov=anomaly_match tests/`).

## Continuous integration

Only pull requests that pass all CI checks will be merged. CI runs:

- License-header check
- Ruff lint + format check
- Dead-code (Vulture) check
- Test suite

Please make sure these pass locally first — CI is not a substitute for running
the checks yourself.

## Code of conduct

We want AnomalyMatch to be a welcoming, respectful, and inclusive space for
everyone. Please be kind and constructive, assume good intent, and treat fellow
contributors with courtesy regardless of their background or experience.
Harassment or discriminatory behaviour of any kind will not be tolerated.

## Questions

Not sure about something? Open an issue — we are happy to help.
