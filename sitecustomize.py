"""Auto-enable Tunix Accel patches when this package is installed.

Python's `site` module imports `sitecustomize` automatically at interpreter
startup when the module is present on `sys.path`. Keep this file tiny and
fail-closed so unrelated Python commands in the same environment are not broken
by optional Tunix/JAX dependencies.
"""

try:
  import tunix_accel.autopatch  # noqa: F401
except Exception:
  pass
