# Releasing

Releases go to PyPI from GitHub Actions through
[trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI trusts this repository's
`publish.yml` workflow, so no API token is stored anywhere.

## One-time setup

Already done for `rubicon-agents`; this section is only needed for a fork or a rename.

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

## Each release

1. Set `version` in `pyproject.toml` and add a `## X.Y.Z - date` section to `CHANGELOG.md`
   (moving any `## Unreleased` notes into it), in a pull request. Merge it once CI is green.
2. Tag the merge commit and publish a GitHub release whose notes are that CHANGELOG section:

   ```bash
   git tag -a vX.Y.Z -m X.Y.Z && git push origin vX.Y.Z
   gh release create vX.Y.Z --verify-tag --title X.Y.Z --notes-file notes.md
   ```

3. Publishing the release starts `publish.yml`. It checks that the tag matches the package
   version, builds the sdist and wheel, runs `twine check` and the test suite against the built
   wheel, then (after your approval, if the environment requires it) uploads to PyPI.

A failed upload can be re-run from the Actions tab. PyPI never accepts the same version twice,
so a broken release is fixed with a new version, not a re-upload.
