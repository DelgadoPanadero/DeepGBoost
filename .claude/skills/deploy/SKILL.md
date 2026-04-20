---
name: deploy
description: End-to-end deployment workflow for DeepGBoost — commit staged changes, open a PR, wait for CI, squash-merge, pull main, bump version (patch/minor/major), tag the release, and publish a GitHub release with auto-generated notes. Trigger on phrases like "deploy", "release", "desplegar", "crear release", "publicar versión", "publish", "cut a release", "ship it", or when the user asks to bump the version and push to PyPI.
version: 1.0.0
---

# Deploy

Full release workflow for `delgadopanadero/deepgboost`. Execute the steps below **in order**. Do not skip a step unless explicitly told to. If any step fails, follow the failure instructions for that step before continuing.

---

## Step 1 — Commit and push current changes

1. Run `git status` to confirm there are staged or unstaged changes worth committing. If the working tree is completely clean (no changes at all), skip to Step 2.
2. If there are unstaged changes the user wants included, stage them explicitly by file name — never use `git add -A` or `git add .` blindly.
3. Generate a commit message by inspecting the staged diff:
   ```bash
   git diff --cached
   ```
   Derive a concise message that follows the project's commit style (`type(#issue): description`). Look at recent commits for style reference:
   ```bash
   git log --oneline -10
   ```
4. Commit:
   ```bash
   git commit -m "<generated message>"
   ```
5. Push the current branch to origin:
   ```bash
   git push -u origin HEAD
   ```

If the push is rejected because the remote branch has diverged, stop and notify the user — do not force-push without explicit instruction.

---

## Step 2 — Create pull request

1. Identify the current branch name:
   ```bash
   git rev-parse --abbrev-ref HEAD
   ```
2. Collect the commit history between this branch and `main` to inform the PR body:
   ```bash
   git log main..HEAD --oneline
   git diff main...HEAD --stat
   ```
3. Compose a PR title (under 70 characters) and a body with sections `## Summary` (bullet list of changes) and `## Test plan` (checklist). Base both on the commit history you collected.
4. Create the PR targeting `main`:
   ```bash
   gh pr create \
     --repo delgadopanadero/deepgboost \
     --base main \
     --title "<generated title>" \
     --body "$(cat <<'EOF'
   ## Summary
   <bullet points derived from commit history>

   ## Test plan
   - [ ] CI checks pass
   - [ ] Behaviour verified locally

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   EOF
   )"
   ```
5. Capture the PR URL printed by `gh pr create` — you will need it in the next step.

---

## Step 3 — Wait for CI

Run the watch command, which blocks until all checks finish:
```bash
gh pr checks --repo delgadopanadero/deepgboost --watch <PR-number-or-URL>
```

Derive the PR number from the URL captured in Step 2 (it is the last path segment).

### If CI passes

All checks show `pass`. Continue to Step 4.

### If CI fails

1. Report to the user exactly which check(s) failed and their log URL.
2. Close the PR:
   ```bash
   gh pr close <PR-number> --repo delgadopanadero/deepgboost --comment "Closing: CI failed on check(s): <failed check names>. Fix and re-run the deploy workflow."
   ```
3. Stop the workflow. Do not attempt to merge or tag anything. Ask the user to fix the failing checks and restart from Step 1.

---

## Step 4 — Squash-merge the PR

Enable squash-merge with auto-merge so GitHub merges as soon as all required checks have passed (they already have at this point):
```bash
gh pr merge <PR-number> \
  --repo delgadopanadero/deepgboost \
  --squash \
  --auto \
  --subject "<one-line summary of the PR — derived from the PR body's Summary section>" \
  --body "<multi-line body — paste the PR body Summary bullet points>"
```

The `--subject` flag sets the squash commit title. Keep it under 72 characters and follow the project commit style (`type(#issue): description`).

Wait for the merge to complete by polling the PR state:
```bash
gh pr view <PR-number> --repo delgadopanadero/deepgboost --json state --jq '.state'
```
Repeat until the value is `MERGED`. Poll at a reasonable interval (do not busy-loop — run the command, read the result, and if not merged, wait a moment before retrying).

---

## Step 5 — Pull main

Switch to `main` and pull the squash-merge commit:
```bash
git checkout main
git pull origin main
```

Verify the latest commit on `main` matches the squash commit subject from Step 4:
```bash
git log --oneline -3
```

---

## Step 6 — Determine version bump

Ask the user:

> "What kind of release is this? Reply with **patch**, **minor**, or **major**."
>
> Current version: read from `pyproject.toml` (see below how to find it).
> - **patch** → increment the third segment (bug fixes, docs, chores). Example: `0.3.1` → `0.3.2`
> - **minor** → increment the second segment, reset patch to 0 (new backward-compatible features). Example: `0.3.1` → `0.4.0`
> - **major** → increment the first segment, reset minor and patch to 0 (breaking changes). Example: `0.3.1` → `1.0.0`

### How to find the current version

Read line 7 of `pyproject.toml` (the `[project]` section's `version` field):
```bash
grep -n '^version' /home/thinbaker/Workspace/DeepGBoost/pyproject.toml
```
The value is in the form `version = "X.Y.Z"`.

### How to compute the new version

Parse X, Y, Z as integers from the current version string, then:
- patch: `X.Y.(Z+1)`
- minor: `X.(Y+1).0`
- major: `(X+1).0.0`

---

## Step 7 — Update pyproject.toml

Edit the `version` field inside `pyproject.toml`. The field is under the `[project]` table and looks exactly like:
```
version = "0.3.1"
```
Replace the old version string with the new one. Make the edit using a precise string replacement — change only that line, nothing else in the file.

After editing, verify the change:
```bash
grep '^version' /home/thinbaker/Workspace/DeepGBoost/pyproject.toml
```
Confirm it shows the new version before continuing.

---

## Step 8 — Commit the version bump to main

Stage only `pyproject.toml`:
```bash
git add /home/thinbaker/Workspace/DeepGBoost/pyproject.toml
```

Commit with a message following the project style:
```bash
git commit -m "chore(): bump version to X.Y.Z"
```

where `X.Y.Z` is the new version determined in Step 6.

---

## Step 9 — Create and push the git tag

Create an annotated tag on the version bump commit:
```bash
git tag -a "vX.Y.Z" -m "Release vX.Y.Z"
```

Push the tag to origin:
```bash
git push origin "vX.Y.Z"
```

Also push the `main` branch so the version bump commit is on the remote:
```bash
git push origin main
```

Verify the tag exists on the remote:
```bash
gh api repos/delgadopanadero/deepgboost/git/refs/tags --jq '.[].ref' | grep "vX.Y.Z"
```

---

## Step 10 — Create the GitHub release

Use `--generate-notes` so GitHub automatically compiles the changelog from commits since the previous tag:
```bash
gh release create "vX.Y.Z" \
  --repo delgadopanadero/deepgboost \
  --title "vX.Y.Z" \
  --generate-notes \
  --latest
```

`--generate-notes` uses GitHub's release notes generator, which groups commits between the previous semver tag and `vX.Y.Z` into categories (features, bug fixes, etc.) based on PR labels and commit messages.

After the command succeeds, capture and display the release URL to the user.

---

## Summary of commands (reference)

| Step | Command |
|------|---------|
| 1 | `git diff --cached`, `git commit`, `git push -u origin HEAD` |
| 2 | `gh pr create --repo delgadopanadero/deepgboost --base main ...` |
| 3 | `gh pr checks --watch <PR>` |
| 4 | `gh pr merge <PR> --squash --auto ...` |
| 5 | `git checkout main && git pull origin main` |
| 6 | Ask user: patch / minor / major |
| 7 | Edit `pyproject.toml` version field |
| 8 | `git add pyproject.toml && git commit -m "chore(): bump version to X.Y.Z"` |
| 9 | `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z && git push origin main` |
| 10 | `gh release create vX.Y.Z --generate-notes --latest` |

---

## Abort conditions

Stop the workflow immediately and notify the user if any of the following occur:

- The push in Step 1 is rejected due to a diverged remote branch.
- The PR cannot be created because the branch has no commits ahead of `main`.
- Any CI check fails in Step 3 (close the PR, report which checks failed).
- The squash-merge in Step 4 is rejected (e.g., branch protection rule not satisfied).
- The tag already exists on the remote — ask the user whether to use a different version string before proceeding.
- `pyproject.toml` does not contain a `version = "..."` line in the expected format — stop and ask the user to inspect the file manually.
