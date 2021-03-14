from setuptools import setup

setup(
    name="kgbench",
    version="0.1",
    description="A set of benchmark datasets for knowledge graph node classification",
    url="",
    author="Peter Bloem (Vrije Universiteit), Xander Wilcke, Lucas van Berkel, Victor de Boer",
    author_email="kgbench@peterbloem.nl",
    packages=["kgbench"],
    install_requires=[
        "numpy",
        "torch",
        "rdflib",
        "hdt"
    ],
    zip_safe=False
)
