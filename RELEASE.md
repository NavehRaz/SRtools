# Release Guide

SRtools is published to PyPI as `srtools-aging`. The Python import package
remains `SRtools`.

## Local validation

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

For a clean install test:

```bash
python -m venv /tmp/srtools-release-test
/tmp/srtools-release-test/bin/python -m pip install dist/srtools_aging-0.1.0-py3-none-any.whl
/tmp/srtools-release-test/bin/python -c "import SRtools; from SRtools import presets; print(presets.get_preset_names()[:3])"
```

## First TestPyPI release

```bash
python -m twine upload --repository testpypi dist/*
```

Then verify installation from TestPyPI in a clean environment:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ srtools-aging==0.1.0
```

## PyPI Trusted Publishing

After the TestPyPI validation works:

1. Create or claim the `srtools-aging` project on PyPI.
2. In PyPI, add a trusted publisher for this GitHub repository.
3. Use workflow name `Release`.
4. Use environment name `pypi`.
5. Protect the `pypi` GitHub environment if desired.

Publish a real release by pushing a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `.github/workflows/release.yml` workflow builds and publishes the package.

## Branch protection

In GitHub repository settings, protect `main`:

- Require a pull request before merging.
- Require your approval before merge.
- Require CI to pass before merge.
- Disallow force pushes.
- Disallow branch deletion.

Public users can open pull requests, but these settings prevent their changes
from entering `main` without approval.
