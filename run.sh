#!/bin/bash
# run.sh — build and run via xcodebuild (Metal shaders can't compile via swift build)
set -e
SCHEME="scry-cli"

echo "Building $SCHEME..."
xcodebuild build \
  -scheme "$SCHEME" \
  -destination 'platform=macOS,arch=arm64' \
  -skipMacroValidation \
  -quiet 2>&1 | tail -5

# Find the binary — use -showBuildSettings and handle the path robustly
BUILD_DIR=$(xcodebuild -scheme "$SCHEME" -showBuildSettings 2>/dev/null \
  | grep ' BUILT_PRODUCTS_DIR' | head -1 | sed 's/.*= //')

if [ -z "$BUILD_DIR" ]; then
  # Fallback: search DerivedData
  BUILD_DIR=$(find ~/Library/Developer/Xcode/DerivedData/scry-*/Build/Products/Debug -maxdepth 0 2>/dev/null | head -1)
fi

BINARY="$BUILD_DIR/$SCHEME"

if [ ! -f "$BINARY" ]; then
  echo "Error: Could not find built binary at $BINARY"
  echo "Try: xcodebuild -scheme $SCHEME -showBuildSettings | grep BUILT_PRODUCTS_DIR"
  exit 1
fi

echo "Running: $BINARY $@"
exec "$BINARY" "$@"
