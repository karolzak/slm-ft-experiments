from setuptools import find_packages, setup


setup(
    name="slm-ft-experiments",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
