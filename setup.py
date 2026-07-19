from pathlib import Path
import ast
import os
import re

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def get_package_version() -> str:
    init_py = (ROOT / "causal_conv1d" / "__init__.py").read_text()
    match = re.search(r"^__version__\s*=\s*(.*)$", init_py, re.MULTILINE)
    version = ast.literal_eval(match.group(1))
    local = os.environ.get("CAUSAL_CONV1D_LOCAL_VERSION")
    return f"{version}+{local}" if local else version


setup(
    name="causal_conv1d",
    version=get_package_version(),
    packages=find_packages(exclude=("tests", "csrc", "rocm_patch")),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Causal depthwise conv1d in CuTe DSL with a PyTorch interface",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/causal-conv1d",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    python_requires=">=3.10",
    install_requires=["torch", "nvidia-cutlass-dsl"],
)
