#!/bin/sh
# One-time: create the labels the automation needs but cannot create itself.
# Run once per repo, then forget it.  Usage: .github/kaino/bootstrap-labels.sh
#
# Not needed for:
#   duplicate   - a GitHub default label, already present
#   needs-issue - pr-issue-link.js creates it when ENFORCE=true
#   size/*      - kaino-pr-hygiene.yml creates it with `gh label create --force`
set -e
REPO="${1:-Kainotomic/omnigent}"

# The only unconditional opt-out from the issue-link rule. Applying it needs
# write access, which is the point -- it must not be self-service.
gh label create skip-issue-check --repo "$REPO" --color ededed \
  --description "Exempt this PR from the every-PR-needs-an-issue rule" --force

echo "Labels ready on $REPO."
