#!/usr/bin/env bash
# The toolchain image's tag. One copy, because two workflows ask for it, and
# the gitlink is in it because the model is built from the submodule.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
{ cat setup/versions.env setup/system.sh setup/restore.sh setup/Dockerfile setup/patches/*/*
  git rev-parse HEAD:third_party/vcix-accelerator
} | sha256sum | cut -c1-12
