from setuptools import setup

setup(
    name="kgbench",
    version="0.1",
    description="A set of benchmark datasets for knowledge graph node classification",
    url="",
    author="Peter Bloem (Vrije Universiteit)",
    author_email="kgbench@peterbloem.nl",
    packages=["kgbench"],
    install_requires=[
        "numpy",
        "torch",
        "rdflib",
    ],
    zip_safe=False
)
