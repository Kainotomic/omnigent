#!/bin/sh
# Merge upstream, always dropping its CI workflows.
#
# Upstream ships ~100 workflows this fork does not run. They are deleted on the
# fork's main, so every upstream merge that touches one produces a modify/delete
# conflict; `git rm` is how that conflict resolves to "stays deleted". The
# pathspec exclusion spares this fork's own kaino-* workflows.
#
# `.gitattributes merge=ours` cannot do this: merge drivers only run when the
# file exists on both sides, never on modify/delete.
set -e

git fetch upstream
git merge --no-edit upstream/main || true

git rm -rq --ignore-unmatch -- '.github/workflows' ':!.github/workflows/kaino-*'

if git diff --name-only --diff-filter=U | grep -q .; then
  echo "Conflicts outside .github/workflows remain:"
  git diff --name-only --diff-filter=U
  echo "Resolve them, then: git commit"
  exit 1
fi

if git diff --cached --quiet; then
  echo "Already up to date."
else
  git commit --no-edit -m "chore: sync upstream (CI workflows dropped)"
fi
