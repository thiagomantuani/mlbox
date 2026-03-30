from setuptools import setup, find_packages

setup(
    name="mlbox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "tqdm>=4.64",
    ],
)