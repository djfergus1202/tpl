
from setuptools import setup, find_packages
setup(
    name="pharma-lab-suite",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    description="Research suite: data analysis, molecular modeling, topological maps, paper generation",
    install_requires=[],
)
