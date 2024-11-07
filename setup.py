from setuptools import setup

setup(
    name="DPMs-ANT",
    py_modules=["ANT"],
    install_requires=[
        "torch",
        "numpy",
        "six~=1.16.0",
        "Pillow~=9.2.0",
        "timm~=0.6.12",
        "termcolor~=1.1.0",
        "PyYAML~=6.0",
        "yacs~=0.1.8",
        "numpy~=1.24.1",
        "einops~=0.6.1",
        "scipy~=1.10.1",
    ],
)
