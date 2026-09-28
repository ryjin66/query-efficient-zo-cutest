#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUTEST_RUNTIME_PREFIX="${CUTEST_RUNTIME_PREFIX:-$HOME/.local/share/cutest-query-benchmark}"
case "$CUTEST_RUNTIME_PREFIX" in *" "*) echo "Choose a runtime path without spaces." >&2; exit 2 ;; esac
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "The supplied compiler lock targets macOS arm64. See README for other platforms." >&2
  exit 2
fi
command -v micromamba >/dev/null || { echo "Install micromamba and add it to PATH." >&2; exit 2; }
xcrun --show-sdk-path >/dev/null
python3 "$ROOT/scripts/verify_release.py" --root "$ROOT" --environment-only
mkdir -p "$CUTEST_RUNTIME_PREFIX/src" "$CUTEST_RUNTIME_PREFIX/opt" "$CUTEST_RUNTIME_PREFIX/mastsif" "$CUTEST_RUNTIME_PREFIX/pycutest_cache"
if [ ! -x "$CUTEST_RUNTIME_PREFIX/conda/bin/python" ]; then
  micromamba create --no-rc -y -p "$CUTEST_RUNTIME_PREFIX/conda" --file "$ROOT/environment/osx-arm64.lock"
fi
if [ ! -x "$CUTEST_RUNTIME_PREFIX/conda/bin/gfortran" ]; then
  ln -s arm64-apple-darwin20.0.0-gfortran "$CUTEST_RUNTIME_PREFIX/conda/bin/gfortran"
fi
source "$ROOT/scripts/activate.sh"
for package in SIFDecode-3.1.0 CUTEst-2.6.0; do
  if [ ! -d "$CUTEST_RUNTIME_PREFIX/src/$package" ]; then
    tar -xzf "$ROOT/environment/$package.tar.gz" -C "$CUTEST_RUNTIME_PREFIX/src"
  fi
done
if [ ! -x "$CUTEST/bin/sifdecoder" ]; then
  meson setup "$CUTEST_RUNTIME_PREFIX/src/SIFDecode-3.1.0/builddir" "$CUTEST_RUNTIME_PREFIX/src/SIFDecode-3.1.0" --prefix "$CUTEST"
  meson compile -C "$CUTEST_RUNTIME_PREFIX/src/SIFDecode-3.1.0/builddir" -j 4
  meson install -C "$CUTEST_RUNTIME_PREFIX/src/SIFDecode-3.1.0/builddir"
fi
if [ ! -f "$CUTEST/lib/libcutest_double.a" ]; then
  meson setup "$CUTEST_RUNTIME_PREFIX/src/CUTEst-2.6.0/builddir" "$CUTEST_RUNTIME_PREFIX/src/CUTEst-2.6.0" --prefix "$CUTEST" -Dmodules=false
  meson compile -C "$CUTEST_RUNTIME_PREFIX/src/CUTEst-2.6.0/builddir" -j 4
  meson install -C "$CUTEST_RUNTIME_PREFIX/src/CUTEst-2.6.0/builddir"
fi
python -m pip install -e "$ROOT[solvers]"
python "$ROOT/scripts/patch_pycutest.py"
tar -xzf "$ROOT/environment/problems.tar.gz" -C "$MASTSIF"
python "$ROOT/scripts/validate_environment.py" --root "$ROOT"
printf 'Ready. In Bash, source "%s/scripts/activate.sh".\n' "$ROOT"
