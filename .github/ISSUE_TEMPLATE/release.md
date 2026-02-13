---
name: New Release
about: Creating a new release version for AnomalyMatch. Only for Maintainers.
title: "Release "
labels: "release"
assignees: "gomezzz"
---

# Feature

## What Needs to Be Done (chronologically)

- [ ] Create PR from `main` -> `develop` to incorporate hotfixes / documentation changes that were made in `main` since the last release.
- [ ] Review the PR (if OK - merge, but DO NOT delete the branch, if problems arise they shall be fixed in `develop` not in `main`)
- [ ] Create a new branch from `develop` called `release` (e.g. `release-0.1.0`)
- [ ] Write changelog into `CHANGELOG.md`
- [ ] Minimize and update packages in `pyproject.toml` based on `environment.yml`
- [ ] Check unit tests -> Check all tests pass on CPU and GPU (e.g. in Datalabs) and that there are tests for all important features
- [ ] Check documentation -> Check presence of documentation for all new or changed user-facing features in README.md
- [ ] Change version number in `pyproject.toml` and `__init__.py`
- [ ] Create PR: `release` → `main`, `release` -> `develop`
- [ ] Double-check that license header workflow passes on the PR
- [ ] Test that you can locally `pip install -e .` the module
- [ ] Request and run PR Review
- [ ] Merge `release` into `main`, and `develop`
- [ ] Create Release on GitHub from the last commit (the one reviewed in the PR) reviewed
- [ ] Download the release from GitHub
- [ ] Prepare for release on the [public GitHub repo](https://github.com/ESA/AnomalyMatch) by removing non-essential folders and files in the downloaded version, i.e. currently `datalabs_setup`, `utility_scripts`
- [ ] Create a PR to the public repo with that version
- [ ] Merge the PR to the public repo
- [ ] Create a new release on the public repo