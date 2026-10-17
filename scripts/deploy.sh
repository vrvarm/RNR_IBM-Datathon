#!/usr/bin/env bash
# Deploy to the LinuxONE server over SSH: pull the latest code from GitHub and (re)start the service.
#
#   DEPLOY_HOST=user@your-linuxone-host ./scripts/deploy.sh
#
# Optional: REMOTE_DIR (default ~/fraud-guard), BRANCH (default main), REPO_URL (default: this repo's origin).
# The server must be able to `git clone` the repo (public repo, or a deploy key on the server).
# Uses docker compose if the server has docker, otherwise a Python venv + nohup.
set -euo pipefail

: "${DEPLOY_HOST:?set DEPLOY_HOST, e.g. DEPLOY_HOST=user@linuxone.example.com}"
REMOTE_DIR="${REMOTE_DIR:-~/fraud-guard}"
BRANCH="${BRANCH:-main}"
REPO_URL="${REPO_URL:-$(git config --get remote.origin.url)}"

ssh "$DEPLOY_HOST" bash -s <<EOF
set -euo pipefail
if [ -d $REMOTE_DIR/.git ]; then
  cd $REMOTE_DIR
  git fetch origin
  git checkout $BRANCH
  git pull --ff-only origin $BRANCH
else
  git clone --branch $BRANCH $REPO_URL $REMOTE_DIR
  cd $REMOTE_DIR
fi
mkdir -p data models

if command -v docker >/dev/null 2>&1; then
  docker compose up -d --build
else
  [ -d .venv ] || python3 -m venv .venv
  . .venv/bin/activate
  pip install -q -r requirements.txt
  [ -f .pid ] && kill \$(cat .pid) 2>/dev/null || true
  nohup scripts/start.sh > service.log 2>&1 &
  echo \$! > .pid
fi
sleep 5
curl -fsS http://localhost:8000/health && echo
EOF

echo "Deployed. API docs: http://${DEPLOY_HOST#*@}:8000/docs"
