# Contributing

## Set up and run the tests

```bash
git clone https://github.com/QoroQuantum/pennylane-maestro.git
cd pennylane-maestro
uv venv && uv pip install -e ".[cloud]" pytest   # the cloud extra needs Python 3.11+
pytest
```

The Qoro cloud tests (`tests/test_divi_backend.py`) run offline: a fake HTTP
transport stands in for the cloud. Without Divi installed they are skipped.

## CI

The Tests workflow runs on every pull request into `main` and every push to it:

- the tests on Python 3.10 to 3.13;
- the Qoro cloud tests with the `cloud` extra;
- a build of the sdist and wheel, checked by `scripts/verify_dist.py`: the sdist
  must contain the whole test suite and pass it when extracted, and the wheel must
  install into a fresh environment, import from outside the checkout and register
  the `maestro.qubit` device.

To run it by hand, use "Run workflow" on the Tests workflow in the Actions tab.

## Making a change

Add a line to [CHANGELOG.md](CHANGELOG.md) under "Unreleased".

## Releasing (maintainers)

1. Move the "Unreleased" changelog entries under a new version heading,
   `## [X.Y.Z] - YYYY-MM-DD`, and add the version's compare link at the bottom.
2. Set `version` in `pyproject.toml` to match. It is the only place the version
   is set: `pennylane_maestro.__version__` reads it from the installed package.
3. Publish a GitHub release for the tag `vX.Y.Z` on `main`. The publish workflow
   checks the tag against `pyproject.toml`, runs the tests, builds and verifies
   the distributions, and uploads them to PyPI through trusted publishing. To
   publish by hand, or to dry-run the build without uploading, use "Run workflow"
   on the Publish to PyPI workflow in the Actions tab; a manual publish also
   creates the GitHub release, with the version's changelog section as its notes.
