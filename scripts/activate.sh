# Source from Bash after creating the environment.
export CUTEST_RUNTIME_PREFIX="${CUTEST_RUNTIME_PREFIX:-$HOME/.local/share/cutest-query-benchmark}"
export CONDA_PREFIX="$CUTEST_RUNTIME_PREFIX/conda"
export PATH="$CONDA_PREFIX/bin:$CUTEST_RUNTIME_PREFIX/opt/bin:$PATH"
if [ -d "$CONDA_PREFIX/etc/conda/activate.d" ]; then
  case $- in *u*) _restore_nounset=1; set +u ;; *) _restore_nounset=0 ;; esac
  for _hook in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do
    [ ! -f "$_hook" ] || source "$_hook"
  done
  [ "$_restore_nounset" = 0 ] || set -u
  unset _hook _restore_nounset
fi
export CUTEST="$CUTEST_RUNTIME_PREFIX/opt"
export SIFDECODE="$CUTEST_RUNTIME_PREFIX/opt"
export MASTSIF="$CUTEST_RUNTIME_PREFIX/mastsif"
export PYCUTEST_CACHE="$CUTEST_RUNTIME_PREFIX/pycutest_cache"
export PYTHONPATH="$PYCUTEST_CACHE${PYTHONPATH:+:$PYTHONPATH}"
export LIBRARY_PATH="$CUTEST/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$CUTEST/include:$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FC="$CONDA_PREFIX/bin/gfortran"
if [ "$(uname -s)" = Darwin ]; then
  export CC=/usr/bin/clang CXX=/usr/bin/clang++
  export SDKROOT="$(xcrun --show-sdk-path)"
  export MACOSX_DEPLOYMENT_TARGET="$(sw_vers -productVersion | awk -F. '{print $1"."$2}')"
  export DYLD_LIBRARY_PATH="$CUTEST/lib:$CONDA_PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
