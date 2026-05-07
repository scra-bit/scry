#!/bin/bash
# run.sh — run the MLX CLI from the most recent Xcode/MCP build products.
#
# `mlx-swift` requires compiled Metal shader resources at runtime. In this repo,
# the supported path is to build through Xcode (or the Xcode MCP), then run the
# resulting product from DerivedData with `DYLD_FRAMEWORK_PATH` pointed at the
# build products directory.
set -euo pipefail

PRODUCT="scry-cli"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DERIVED_DATA_ROOT="${XCODE_DERIVED_DATA_ROOT:-$HOME/Library/Developer/Xcode/DerivedData}"

find_latest_build_dir() {
  find "$DERIVED_DATA_ROOT" -path "*/Build/Products/Debug/$PRODUCT" -type f -print 2>/dev/null \
    | while IFS= read -r binary; do
        stat -f "%m %N" "$binary"
      done \
    | sort -nr \
    | sed -n '1s/^[0-9]* //p' \
    | xargs -I{} dirname "{}"
}

BUILD_DIR="${XCODE_PRODUCTS_DIR:-}"
if [ -z "$BUILD_DIR" ]; then
  BUILD_DIR="$(find_latest_build_dir || true)"
fi

if [ -z "$BUILD_DIR" ]; then
  cat <<'EOF'
Error: no Xcode-built scry-cli binary was found.

Build the package first in one of these ways:
  - Xcode UI: build the `scry-Package` scheme for `My Mac`
  - Codex/Xcode MCP: run the project build from the active Xcode window

Then rerun:
  ./run.sh <args>
EOF
  exit 1
fi

if [ ! -x "$BUILD_DIR/$PRODUCT" ]; then
  echo "Error: expected executable at $BUILD_DIR/$PRODUCT"
  exit 1
fi

if [ ! -d "$BUILD_DIR/mlx-swift_Cmlx.bundle" ]; then
  echo "Error: missing MLX resource bundle at $BUILD_DIR/mlx-swift_Cmlx.bundle"
  echo "Rebuild the package through Xcode UI or Xcode MCP, then rerun this script."
  exit 1
fi

echo "Running Xcode-built binary: $BUILD_DIR/$PRODUCT $*"
export DYLD_FRAMEWORK_PATH="$BUILD_DIR${DYLD_FRAMEWORK_PATH:+:$DYLD_FRAMEWORK_PATH}"
exec "$BUILD_DIR/$PRODUCT" "$@"
