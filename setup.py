from setuptools import find_packages, setup

with open("./README.md") as f:
    long_description = f.read()

setup(
    name="retroactive-pq",
    description=("Retroactive priority queues and useful data structures for"
                 + " retroactivity"),
    author="Parker J. Rule",
    author_email="parker.rule@tufts.edu",
    maintainer="Parker J. Rule",
    maintainer_email="parker.rule@tufts.edu",
    long_description=long_description,
    long_description_content_type="text/x-markdown",
    url="https://github.com/6851-2021/retroactive-pq",
    packages=['retroactive_pq'],
    version='0.1',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ])
