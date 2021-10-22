import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pynanz-salemileandro",
    version="0.0.0",
    author="Leandro Salemi",
    author_email="salemileandro@gmail.com",
    description="Python toolkit for financial planning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salemileandro/pynanz",
    packages=setuptools.find_packages(),
    entry_points = {"console_scripts" : ['pynanz=pynanz.cli:main']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=False,
)

