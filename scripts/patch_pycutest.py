#!/usr/bin/env python3
"""Patch PyCUTEst macOS path detection for the local conda-forge toolchain."""

from __future__ import annotations

import sysconfig
from pathlib import Path


SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])
PYCUTEST_DIR = SITE_PACKAGES / "pycutest"


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Could not find expected PyCUTEst patch target in {path}")
    path.write_text(text.replace(old, new, 1))
    return True


def main() -> int:
    system_paths = PYCUTEST_DIR / "system_paths.py"
    install_scripts = PYCUTEST_DIR / "install_scripts.py"
    if not system_paths.exists() or not install_scripts.exists():
        raise RuntimeError(f"PyCUTEst package not found under {PYCUTEST_DIR}")

    changed = False
    changed |= replace_once(
        system_paths,
        """homebrew_prefix = None
if sys.platform == 'darwin':  # Mac
    import subprocess
    homebrew_prefix = subprocess.check_output(['brew', '--prefix']).decode('utf-8')[:-1]
""",
        """homebrew_prefix = None
if sys.platform == 'darwin':  # Mac
    import subprocess
    try:
        homebrew_prefix = subprocess.check_output(['brew', '--prefix']).decode('utf-8')[:-1]
    except (OSError, subprocess.SubprocessError):
        homebrew_prefix = None
""",
    )
    changed |= replace_once(
        system_paths,
        """def get_homebrew_gfortran_path():
    if sys.platform == 'darwin':  # Mac
        gfortran_path = max(glob(homebrew_prefix + '/Cellar/gcc/*/lib/gcc/*/'),key=os.path.getmtime)
        if os.path.isdir(gfortran_path):
            return gfortran_path
        # Raise error if GCC not found
        raise RuntimeError('Could not find gfortran - has gcc been installed and linked with homebrew?')
""",
        """def get_homebrew_gfortran_path():
    if sys.platform == 'darwin':  # Mac
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            candidates = [os.path.join(conda_prefix, 'lib')]
            candidates.extend(glob(os.path.join(conda_prefix, 'lib', 'gcc', '*', '*')))
            for gfortran_path in candidates:
                if os.path.isfile(os.path.join(gfortran_path, 'libgfortran.dylib')) or os.path.isfile(os.path.join(gfortran_path, 'libgfortran.a')):
                    return gfortran_path
        if homebrew_prefix:
            gfortran_path = max(glob(homebrew_prefix + '/Cellar/gcc/*/lib/gcc/*/'),key=os.path.getmtime)
            if os.path.isdir(gfortran_path):
                return gfortran_path
        # Raise error if GCC not found
        raise RuntimeError('Could not find gfortran - has gcc been installed and linked with homebrew or conda?')
""",
    )
    changed |= replace_once(
        install_scripts,
        """libraries=['gfortran']
library_dirs=['%s']
extra_link_args=['-Wl,-no_compact_unwind']
""",
        """libraries=['gfortran', 'quadmath']
library_dirs=['%s']
conda_lib=os.path.join(os.environ.get('CONDA_PREFIX',''), 'lib')
if conda_lib and os.path.isdir(conda_lib):
    library_dirs.append(conda_lib)
extra_link_args=['-Wl,-no_compact_unwind']
for libdir in library_dirs:
    extra_link_args.append('-Wl,-rpath,' + libdir)
""",
    )

    print("Patched PyCUTEst for conda gfortran" if changed else "PyCUTEst conda patch already applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
