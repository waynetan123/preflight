#!/usr/bin/env bash
# Named git worktrees, each running as its own Docker Compose stack (postgres, backend, worker, frontend)
# with a copy of the main database.
#
#   scripts/worktree.sh new <name> [--from <ref>] [--clean] [--no-sources]
#       Creates branch <name> (or checks it out if it exists) in ../SSIP_v3-worktrees/<name>, then starts
#       compose project ssip-<name> on its own ports with a copy of the main stack's database.
#       Without --from, the worktree starts at HEAD and gets your uncommitted changes too (--clean skips them).
#       --no-sources copies the src_* tables (each data source's raw rows, read only by builds) empty: a
#       faster copy, but a build there skips every source until `python -m app.cli load-sources` loads them.
#       SSIP_WT_COPY_JOBS sets how many tables the copy dumps and restores at once (default 4).
#   scripts/worktree.sh rm <name> [--force]
#       Deletes the stack (containers, database volume, images) and the worktree, and the branch if it's
#       merged. --force also throws away uncommitted changes in the worktree.
#   scripts/worktree.sh ls
#
# Inside a worktree, plain `docker compose ...` (stop, start, logs -f backend, up -d --build) targets its stack.
# Needs the main stack's Postgres running: `docker compose up -d postgres`.
set -euo pipefail

MAIN="$(git -C "$(dirname "$0")" worktree list --porcelain | awk 'NR==1 {print $2}')"
WT_ROOT="${SSIP_WORKTREE_ROOT:-$(dirname "$MAIN")/SSIP_v3-worktrees}"
OVERRIDE=compose.worktree.yml

die() { echo "error: $*" >&2; exit 1; }
main_compose() { docker compose -f "$MAIN/docker-compose.yml" "$@"; }

# Folder, compose project and volume names: lowercase [a-z0-9-].
slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed -E 's/-+/-/g; s/^-|-$//g'
}

port_free() { ! lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# Slot n gets ports 3000+n, 8000+n and 5432+n. Skip slots another worktree has claimed (even if its stack is
# stopped) or whose ports are taken.
pick_slot() {
  local n claimed
  claimed="$(cat "$WT_ROOT"/*/.env 2>/dev/null | sed -n 's/^SSIP_WT_SLOT=//p')"
  for n in $(seq 1 99); do
    grep -qx "$n" <<<"$claimed" && continue
    port_free $((3000 + n)) && port_free $((8000 + n)) && port_free $((5432 + n)) && { echo "$n"; return; }
  done
  die "no free port slot"
}

exclude_from_git() {
  local exclude
  exclude="$(git -C "$MAIN" rev-parse --path-format=absolute --git-common-dir)/info/exclude"
  grep -qxF "$1" "$exclude" 2>/dev/null || echo "$1" >> "$exclude"
}

cmd_new() {
  local name="" from="" clean=0 no_sources=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --from) from="$2"; shift 2 ;;
      --clean) clean=1; shift ;;
      --no-sources) no_sources=1; shift ;;
      -*) die "unknown option $1" ;;
      *) [ -z "$name" ] || die "one name only"; name="$1"; shift ;;
    esac
  done
  [ -n "$name" ] || die "usage: worktree.sh new <name> [--from <ref>] [--clean] [--no-sources]"
  git check-ref-format --branch "$name" >/dev/null || die "'$name' isn't a valid branch name"

  local s project path slot fe be pg
  s="$(slug "$name")"; project="ssip-$s"; path="$WT_ROOT/$s"
  [ ! -e "$path" ] || die "$path already exists"
  [ -z "$(docker compose ls -aq --filter "name=^$project\$")" ] || die "compose project $project already exists"
  main_compose exec -T postgres pg_isready -U ssip -q 2>/dev/null \
    || die "the main stack's Postgres isn't running (docker compose up -d postgres)"
  mkdir -p "$WT_ROOT"
  slot="$(pick_slot)"; fe=$((3000 + slot)); be=$((8000 + slot)); pg=$((5432 + slot))

  echo "→ worktree $path"
  if git -C "$MAIN" show-ref --verify --quiet "refs/heads/$name"; then
    [ -z "$from" ] || die "branch $name already exists; drop --from"
    git -C "$MAIN" worktree add "$path" "$name"
  else
    git -C "$MAIN" worktree add -b "$name" "$path" "${from:-HEAD}"
  fi

  if [ -z "$from" ] && [ "$clean" -eq 0 ]; then
    echo "→ carrying over uncommitted changes"
    git -C "$MAIN" diff --binary HEAD | git -C "$path" apply --allow-empty
    (cd "$MAIN" && git ls-files --others --exclude-standard -z | xargs -0 -I{} sh -c \
      'mkdir -p "$1/$(dirname "$2")" && cp -p "$2" "$1/$2"' _ "$path" {})
  fi

  echo "→ compose project $project (frontend :$fe, backend :$be, postgres :$pg)"
  # Compose reads .env in the project directory, so `docker compose` run here picks up both files and the name.
  cat > "$path/.env" <<EOF
COMPOSE_PROJECT_NAME=$project
COMPOSE_FILE=docker-compose.yml:$OVERRIDE
SSIP_WT_SLOT=$slot
EOF
  cat > "$path/$OVERRIDE" <<EOF
# Written by scripts/worktree.sh: this worktree's ports and its own database volume.
services:
  postgres:
    ports: !override
      - "127.0.0.1:$pg:5432"
    # Drops the create-test-db.sql bind mount: OrbStack can't read ~/Desktop without macOS file permission.
    # The script creates ssip_test instead.
    volumes: !override
      - ssip-pg:/var/lib/postgresql/data
  backend:
    ports: !override
      - "127.0.0.1:$be:8000"
  frontend:
    ports: !override
      - "127.0.0.1:$fe:3000"
volumes:
  ssip-pg:
    name: $project-pg
EOF
  exclude_from_git "/$OVERRIDE"

  [ -f "$MAIN/backend/.env" ] || die "no backend/.env in $MAIN"
  # Compose overrides SSIP_DATABASE_URL inside the containers; this value is for pytest/CLI runs from the host.
  sed -E "s#^(SSIP_DATABASE_URL=[^@]*@)[^/]+/[^/?]+#\1localhost:$pg/ssip#" "$MAIN/backend/.env" > "$path/backend/.env"
  [ -d "$MAIN/backend/secrets" ] && cp -R "$MAIN/backend/secrets" "$path/backend/secrets"
  # For editors, tests and type checks run from the host. The images install their own dependencies.
  ln -s "$MAIN/.venv" "$path/.venv"
  exclude_from_git "/.venv" # .gitignore's `.venv/` only matches directories, not this symlink
  [ -d "$MAIN/frontend/node_modules" ] && cp -cR "$MAIN/frontend/node_modules" "$path/frontend/node_modules"

  cd "$path"
  echo "→ starting postgres"
  docker compose up -d --wait postgres
  docker compose exec -T postgres createdb -U ssip -O ssip ssip_test
  copy_database "$project" "$no_sources"
  echo "→ building and starting backend, worker, frontend"
  docker compose up -d --build --wait

  cat <<EOF

Done: $path  (branch $name, compose project $project)

  App       http://127.0.0.1:$fe   (use 127.0.0.1, not localhost, so its login cookie stays separate)
  API docs  http://127.0.0.1:$be/docs
  Postgres  localhost:$pg

From $path:
  docker compose stop | start         spin the stack down / up
  docker compose up -d --build        rebuild after code changes
  docker compose logs -f backend
EOF
  [ "$no_sources" -eq 0 ] || cat <<EOF

The data sources' raw tables (src_*) are empty, so a build here skips every source and keeps their accounts.
To build from a source, load it first (all sources, or those named):
  docker compose exec backend python -m app.cli load-sources [<source> ...] --now
EOF
}

# Copies the main database into this stack's (run from the worktree): a directory-format dump in the main
# Postgres container, streamed across as a tar, then restored. Both sides work on several tables at once, so
# target_accounts' indexes build side by side instead of one after another. With no_sources, the src_* tables
# come across with no rows: a build skips a source whose tables are empty (app/models/raw_tables.py).
copy_database() {
  local project="$1" no_sources="$2" jobs="${SSIP_WT_COPY_JOBS:-4}" dump exclude=() started=$SECONDS
  dump="/tmp/ssip-copy-$project"
  [ "$no_sources" -eq 0 ] || exclude=(--exclude-table-data='src_*')
  echo "→ copying the main database$([ "$no_sources" -eq 0 ] || echo ", without the data sources' rows")"
  # The dump is in both containers until the restore ends; removed on the way out, whether or not it worked.
  trap "main_compose exec -T postgres rm -rf '$dump' >/dev/null 2>&1 || true
        docker compose exec -T postgres rm -rf '$dump' >/dev/null 2>&1 || true" EXIT
  main_compose exec -T postgres rm -rf "$dump"
  main_compose exec -T postgres pg_dump -U ssip -Fd -j "$jobs" --compress=lz4 \
    ${exclude[@]+"${exclude[@]}"} -f "$dump" ssip
  main_compose exec -T postgres tar -C /tmp -cf - "${dump#/tmp/}" | docker compose exec -T postgres tar -C /tmp -xf -
  docker compose exec -T postgres pg_restore -U ssip -d ssip -j "$jobs" --no-owner --exit-on-error "$dump"
  main_compose exec -T postgres rm -rf "$dump"
  docker compose exec -T postgres rm -rf "$dump"
  trap - EXIT
  echo "  copied in $((SECONDS - started))s"
}

cmd_rm() {
  local name="" force=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --force) force=1; shift ;;
      -*) die "unknown option $1" ;;
      *) name="$1"; shift ;;
    esac
  done
  [ -n "$name" ] || die "usage: worktree.sh rm <name> [--force]"
  local s project path branch=""
  s="$(slug "$name")"; project="ssip-$s"; path="$WT_ROOT/$s"

  if [ -d "$path" ]; then
    branch="$(git -C "$path" branch --show-current)"
    if [ "$force" -eq 0 ] && [ -n "$(git -C "$path" status --porcelain)" ]; then
      git -C "$path" status --short
      die "the worktree has uncommitted changes (above); commit them, or rm --force to discard them"
    fi
  fi

  echo "→ deleting compose project $project"
  if [ -f "$path/.env" ]; then
    (cd "$path" && docker compose down -v --rmi local --remove-orphans)
  else
    docker compose -p "$project" down -v --rmi local --remove-orphans 2>/dev/null || true
    docker volume rm "$project-pg" 2>/dev/null || true
  fi

  if [ -d "$path" ]; then
    echo "→ removing worktree $path"
    git -C "$MAIN" worktree remove --force "$path"
  fi
  if [ -n "$branch" ]; then
    if git -C "$MAIN" branch -d "$branch" >/dev/null 2>&1; then
      echo "→ deleted merged branch $branch"
    else
      echo "→ kept branch $branch (not merged; git branch -D $branch to delete it)"
    fi
  fi
}

cmd_ls() {
  local env dir slot
  git -C "$MAIN" worktree list
  echo
  docker compose ls -a --filter "name=^ssip-"
  for env in "$WT_ROOT"/*/.env; do
    [ -f "$env" ] || continue
    dir="$(dirname "$env")"; slot="$(sed -n 's/^SSIP_WT_SLOT=//p' "$env")"
    [ -n "$slot" ] && printf '%s  app http://127.0.0.1:%s  api :%s  postgres :%s\n' \
      "$(basename "$dir")" $((3000 + slot)) $((8000 + slot)) $((5432 + slot))
  done
}

# Claude Code's WorktreeCreate/WorktreeRemove hooks (.claude/settings.json) call these, so `claude -w <name>`,
# EnterWorktree and subagent worktrees all get a stack. Input is JSON on stdin; create must print only the
# worktree path on stdout.
hook_create() {
  local name
  name="$(jq -r '.name // empty')"
  [ -n "$name" ] || die "WorktreeCreate input has no name"
  cmd_new "$name" >&2
  echo "$WT_ROOT/$(slug "$name")"
}

hook_remove() {
  local path
  path="$(jq -r '.worktree_path // empty')"
  case "$path" in
    "$WT_ROOT"/*) ;;
    *) die "won't remove $path: not under $WT_ROOT" ;;
  esac
  # Claude Code has already confirmed the removal with the user.
  cmd_rm "$(basename "$path")" --force >&2
}

case "${1:-}" in
  new) shift; cmd_new "$@" ;;
  rm) shift; cmd_rm "$@" ;;
  ls) cmd_ls ;;
  hook-create) hook_create ;;
  hook-remove) hook_remove ;;
  *) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
