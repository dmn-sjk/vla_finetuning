"""Native-library loading fixes that must run before torch/torchcodec import.

Call `preload_npp()` at the very top of an entrypoint, before importing lerobot
or torchcodec (lerobot imports torchcodec transitively).
"""

import ctypes
import os

# torchcodec's core libraries link NVIDIA NPP for GPU colour conversion, but torch
# ships no NPP — it has no use for it — so nothing puts NPP on the loader path and
# every libtorchcodec_core*.so fails to dlopen with "libnppicc.so.12: cannot open
# shared object file". The nvidia-npp-cu12 wheel provides the libraries; this makes
# the dynamic linker able to see them.
#
# Loading them RTLD_GLOBAL rather than exporting LD_LIBRARY_PATH keeps the fix
# inside the process: the path is derived from the installed wheel, so it is correct
# on any machine that ran `uv sync`, and there is no environment variable for a
# caller to forget. (FFmpeg still needs LD_LIBRARY_PATH — those libraries come from
# a micromamba prefix outside Python and are not discoverable from here.)
_NPP_LIBS = (
    "libnppc.so.12",    # core; the others depend on it, so it must load first
    "libnppicc.so.12",  # colour conversion — the one torchcodec actually names
    "libnppig.so.12",   # geometry (resize/crop)
)


def preload_npp() -> bool:
    """Load NPP into the global symbol namespace. Returns False if unavailable.

    Never raises: on a machine where torchcodec resolves NPP by other means (a
    system CUDA install, or NPP already on LD_LIBRARY_PATH) this is redundant, and
    a missing wheel should surface as torchcodec's own error, not ours.
    """
    try:
        import nvidia.npp
    except ImportError:
        return False

    lib_dir = os.path.join(os.path.dirname(nvidia.npp.__file__), "lib")
    loaded = False
    for so in _NPP_LIBS:
        path = os.path.join(lib_dir, so)
        if not os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded = True
        except OSError:
            pass
    return loaded
