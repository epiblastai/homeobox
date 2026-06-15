#!/usr/bin/env bash
set -euo pipefail

repo="${AUTO_ATLAS_REPO:-epiblastai/auto-atlas}"
ref="${AUTO_ATLAS_REF:-main}"
agent_skills_dir="${AGENT_SKILLS_DIR:-"$HOME/.agents/skills"}"
claude_skills_dir="${CLAUDE_SKILLS_DIR:-"$HOME/.claude/skills"}"

for cmd in curl tar mktemp; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

curl -fsSL "https://codeload.github.com/$repo/tar.gz/$ref" -o "$tmp_dir/repo.tar.gz"
tar -xzf "$tmp_dir/repo.tar.gz" -C "$tmp_dir" --strip-components=1

skill_count=0
for skill_dir in "$tmp_dir"/skills/*; do
  if [ ! -d "$skill_dir" ] || [ ! -f "$skill_dir/SKILL.md" ]; then
    continue
  fi

  skill_name="${skill_dir##*/}"
  mkdir -p "$agent_skills_dir" "$claude_skills_dir"
  rm -rf "$agent_skills_dir/$skill_name" "$claude_skills_dir/$skill_name"
  cp -R "$skill_dir" "$agent_skills_dir/"
  ln -s "$agent_skills_dir/$skill_name" "$claude_skills_dir/$skill_name"

  skill_count=$((skill_count + 1))
done

if [ "$skill_count" -eq 0 ]; then
  echo "No skills found in $repo@$ref" >&2
  exit 1
fi

echo "Installed $skill_count auto-atlas skills to $agent_skills_dir"
echo "Linked Claude skills in $claude_skills_dir"
