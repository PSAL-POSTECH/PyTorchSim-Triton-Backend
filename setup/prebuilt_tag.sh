#!/usr/bin/env bash
# The prebuilt image's tag: the toolchain tag it is FROM, the steps it runs, its stages in the Dockerfile,
# what those steps build -- the vcix-accelerator submodule's tree and the files the prebuilt stage copies --
# and the torch PyTorchSim's bundle names (its bundle.json, less the commits a new bundle moves).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -z "$(git -C third_party/vcix-accelerator rev-parse --show-prefix 2>/dev/null || echo none)" ] || {
  echo "prebuilt_tag.sh: third_party/vcix-accelerator is not a checked-out submodule; run git submodule update --init third_party/vcix-accelerator" >&2
  exit 1
}
code() { grep -hvE '^\s*(#|$)' "$@"; }
stages=$(sed -n '/^# =* prebuilt =*$/,/^# =* compiler =*$/p' setup/Dockerfile)
[ -n "$stages" ] || { echo "prebuilt_tag.sh: setup/Dockerfile has no prebuilt section" >&2; exit 1; }
P=pytorchsim_triton_compiler
{ setup/toolchain_tag.sh
  code setup/toolchain.sh setup/versions.env
  code <<<"$stages"
  git -C third_party/vcix-accelerator rev-parse 'HEAD^{tree}'
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["torch"])' validation/pytorchsim/bundle.json
  for path in $P/__init__.py $P/machine.py $P/toolchain.py $P/contract/__init__.py $P/contract/target.py \
              $P/dialects/src $P/triton_extension/CMakeLists.txt $P/triton_extension/include $P/triton_extension/lib; do
    git rev-parse "HEAD:$path"
  done
} | sha256sum | cut -c1-12
