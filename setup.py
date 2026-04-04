from setuptools import setup, find_packages

setup(
    name="pennylane-maestro",
    version="0.1.0",
    description="PennyLane plugin for the Maestro quantum simulator by Qoro Quantum",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Qoro Quantum",
    author_email="team@qoroquantum.de",
    url="https://github.com/QoroQuantum/pennylane-maestro",
    license="GPL-3.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pennylane_maestro": ["config.toml"],
    },
    install_requires=[
        "pennylane>=0.38",
        "qoro-maestro>=0.2.8",
        "numpy",
    ],
    entry_points={
        "pennylane.plugins": [
            "maestro.qubit = pennylane_maestro:MaestroQubitDevice",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
