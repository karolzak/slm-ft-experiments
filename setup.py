from setuptools import find_packages, setup


setup(
    name="slm-ft-experiments",
    version="0.1.0",
    description="Initial scaffolding for SLM finetuning experiments",
    author="karolzak",
    python_requires=">=3.10",
    install_requires=["python-dotenv>=1.0.0"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
