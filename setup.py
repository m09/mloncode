from importlib.machinery import SourceFileLoader
from pathlib import Path
from sys import stderr
from types import ModuleType

from setuptools import find_packages, setup

try:
    import torch  # noqa: F401
    import dgl  # noqa: F401
except ImportError:
    print(
        "PyTorch and DGL should be installed. "
        "Please visit https://pytorch.org/ and https://www.dgl.ai/ for instructions.",
        file=stderr,
    )

loader = SourceFileLoader("mloncode", "./mloncode/__init__.py")
mloncode = ModuleType(loader.name)
loader.exec_module(mloncode)

setup(
    name="mloncode",
    version=mloncode.__version__,  # type: ignore
    description="ML on Code.",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Machine Learning on Code Organization",
    author_email="contact@mlonco.de",
    python_requires=">=3.6.0",
    url="https://github.com/mloncode/mloncode",
    packages=find_packages(exclude=["tests"]),
    entry_points={"console_scripts": ["mloncode=mloncode.__main__:main"]},
    install_requires=[
        "coloredlogs",
        "bblfsh <3.0",
        "asdf",
        "dulwich",
        "matplotlib",
        "pytorch-lightning",
        "tensorflow",
        "Pillow",
    ],
    extras_require=dict(
        test=["codecov", "pytest"],
        dev=[
            "flake8",
            "flake8-bugbear",
            "flake8-docstrings",
            "flake8-import-order",
            "pylint",
            "mypy",
            "black",
        ],
    ),
    include_package_data=True,
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha"
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Pre-processors",
        "Topic :: Software Development :: Quality Assurance",
    ],
)
