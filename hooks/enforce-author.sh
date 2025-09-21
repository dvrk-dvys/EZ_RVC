#!/usr/bin/env bash
set -euo pipefail

# ✅ EDIT THESE to your allowed identity(ies)
allowed_name_regex='^(dvrk-dvys)$'
allowed_email_regex='^(ja\.harr91@gmail\.com)$'

author_ident="$(git var GIT_AUTHOR_IDENT || true)"
committer_ident="$(git var GIT_COMMITTER_IDENT || true)"

# Parse "Name <email>"
parse_name() { echo "${1%% <*}"; }
parse_email() { local x="${1#*<}"; echo "${x%>*}"; }

author_name="$(parse_name "$author_ident")"
author_email="$(parse_email "$author_ident")"
committer_name="$(parse_name "$committer_ident")"
committer_email="$(parse_email "$committer_ident")"

lower_blob="$(printf '%s%s' "$author_ident" "$committer_ident" | tr '[:upper:]' '[:lower:]')"

# Hard-block anything mentioning "claude"
if echo "$lower_blob" | grep -qi 'claude'; then
  echo "❌ Commits authored by or via 'Claude' are not allowed."
  exit 1
fi

fail() {
  echo "❌ $1 must be you. Found: name='$2', email='$3'"
  echo "   Fix locally with:"
  echo "     git config user.name 'dvrk-dvys'"
  echo "     git config user.email 'ja.harr91@gmail.com'"
  exit 1
}

# Use POSIX-compatible regex checks
echo "$author_name"    | grep -Eq "$allowed_name_regex"  || fail "Author"    "$author_name"    "$author_email"
echo "$author_email"   | grep -Eq "$allowed_email_regex" || fail "Author"    "$author_name"    "$author_email"
echo "$committer_name" | grep -Eq "$allowed_name_regex"  || fail "Committer" "$committer_name" "$committer_email"
echo "$committer_email"| grep -Eq "$allowed_email_regex" || fail "Committer" "$committer_name" "$committer_email"
