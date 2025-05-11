from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="light-gen",
    version="0.0.12",
    description="Generating lightcones from IllustrisTNG simulations",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SoumyaShreeram/LightGen",
    author="SoumyaShreeram",
    author_email="shreeramsoumya@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["astropy >= 5.2.2", "seaborn >= 0.12.2", "numpy >= 1.24.3", "scipy >= 1.10.1", "h5py >= 3.10.0", "pandas >= 2.0.1"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9",
)