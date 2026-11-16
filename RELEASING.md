# Releasing

Releases go to PyPI from GitHub Actions through
[trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI trusts this repository's
`publish.yml` workflow, so no API token is stored anywhere.

## One-time setup

1. On PyPI, sign in and open **Your account > Publishing**. Under **Add a new pending
   publisher**, choose **GitHub** and enter:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `rubicon-agents` |
   | Owner | `Ashin-06` |
   | Repository name | `rubicon` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   The pending publisher turns into a normal one on the first upload, which also creates the
   project.
2. On GitHub, open the repository's **Settings > Environments**, create an environment named
   `pypi`, and (recommended) add yourself under **Required reviewers**, so every upload waits
   for your approval.

Optionally repeat both steps on [TestPyPI](https://test.pypi.org) to rehearse.

## Each release

1. Set `version` in `pyproject.toml` and move the `Unreleased` section of `CHANGELOG.md` under
   the new version, in a pull request. Merge it once CI is green.
2. Tag the merge commit and publish a GitHub release whose notes are that CHANGELOG section:

   ```bash
   git tag -a v0.10.0 -m 0.10.0 && git push origin v0.10.0
   gh release create v0.10.0 --verify-tag --title 0.10.0 --notes-file notes.md
   ```

3. Publishing the release starts `publish.yml`. It checks that the tag matches the package
   version, builds the sdist and wheel, runs `twine check` and the test suite against the built
   wheel, then (after your approval, if the environment requires it) uploads to PyPI.

A failed upload can be re-run from the Actions tab. PyPI never accepts the same version twice,
so a broken release is fixed with a new version, not a re-upload.
