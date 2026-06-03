"""Fail-closed startup import for tunix-accel autopatches."""

try:
  import tunix_accel.autopatch  # noqa: F401
except Exception:
  pass
