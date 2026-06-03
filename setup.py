import os

from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
  """Copy the startup .pth file to the wheel/editable top level."""

  def run(self):
    super().run()
    self.copy_file(
        "tunix_accel_autopatch.pth",
        os.path.join(self.build_lib, "tunix_accel_autopatch.pth"),
    )


setup(
    packages=find_packages(include=["tunix_accel", "tunix_accel.*"]),
    py_modules=["sitecustomize", "tunix_accel_startup"],
    cmdclass={"build_py": build_py},
)
