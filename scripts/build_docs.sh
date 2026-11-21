#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

for docs_tool in doxygen zensical python3; do
  if ! command -v "$docs_tool" >/dev/null 2>&1; then
    printf '%s is required to build the documentation.\n' "$docs_tool" >&2
    exit 1
  fi
done

# Remove obsolete API pages when a public symbol is renamed or removed.
rm -rf -- build-docs/doxygen docs/reference
mkdir -p -- build-docs/doxygen docs/reference
for api_reference in capture replay; do
  doxygen "docs/doxygen/${api_reference}.doxyfile"
  python3 scripts/render_api.py "build-docs/doxygen/${api_reference}/xml" \
    "docs/reference/${api_reference}.md"
done
zensical build --clean --strict
