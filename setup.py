import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chatbot-bmcandrews",
    version="0.0.1",
    author="Barry McAndrews",
    author_email="bmcandrews@pitt.edu",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bmcandrews/chatbot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
    install_requires=[
        'keras',
        'tensorflow',
        'numpy',
    ],
)
