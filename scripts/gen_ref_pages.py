#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Auto-generate API reference pages by walking the package tree.

Run this before ``zensical build``/``serve``: Zensical has no equivalent of the
``mkdocs-gen-files`` plugin, so the pages are written to ``docs/api/`` on disk
(gitignored) instead of being materialised during the build.
"""

import shutil
from pathlib import Path

# Anchored to the script, not the cwd — run from elsewhere and a cwd-relative
# glob would silently match nothing and ship a site with no API reference
REPO_ROOT = Path(__file__).resolve().parent.parent

PACKAGES = ["anomaly_match", "anomaly_match_ui"]


def main() -> None:
    """Regenerate ``docs/api`` from the docstrings of every documented package.

    Raises:
        FileNotFoundError: If a package in ``PACKAGES`` is missing, e.g. after a rename.
        RuntimeError: If no page was written, which would publish an empty reference.
    """
    api_dir = REPO_ROOT / "docs" / "api"

    # Wipe first so modules deleted since the last run don't linger as stale pages
    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for package in PACKAGES:
        package_path = REPO_ROOT / package
        if not package_path.is_dir():
            raise FileNotFoundError(f"Documented package not found: {package_path}")

        for path in sorted(package_path.rglob("*.py")):
            relative_path = path.relative_to(REPO_ROOT)
            if path.name.startswith("_") or any(
                part.startswith(".") for part in relative_path.parts
            ):
                continue

            doc_path = api_dir / relative_path.with_suffix(".md")
            doc_path.parent.mkdir(parents=True, exist_ok=True)

            identifier = ".".join(relative_path.with_suffix("").parts)
            with open(doc_path, "w", encoding="utf-8") as fd:
                # Explicit title: Zensical would otherwise prettify the file name for
                # the navigation, turning `checkpoint_io` into "Checkpoint io"
                fd.write(f"---\ntitle: {path.stem}\n---\n\n::: {identifier}\n")
            written += 1

    if written == 0:
        raise RuntimeError(f"No API pages generated from {PACKAGES}")

    print(f"{written} API pages written to {api_dir}")


if __name__ == "__main__":
    main()
