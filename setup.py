from setuptools import setup, find_packages

setup(
    name="SepMe",
    version="1.2",
    packages=find_packages(),
    install_requires=[
        "mlflow",
        "sklearn",
        "scipy",
        "pandas",
        "numpy",
        "click",
        "nglpy",
        "networkx",
        "psutil",
        "yaml",
        "shapely",
        "descartes",
        "matplotlib",
        "seaborn",
        "umap",
        "six",
    ],
)
