from setuptools import find_packages, setup

setup(
    name="rosettas_repo",
    version="0.1.0",
    author="Giovanni Campa",
    author_email="giocampa93@gmail.com",
    description="Implementation of various ML algorithms with pytorch and keras",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/yourusername/your_project_name",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.0",
        "tensorflow==2.15",
        "scikit-learn==1.4.0",
        "torch==2.2.1",
        "torchvision==0.17.1",
        "matplotlib==3.8.2",
        "black==24.2.0",
        "isort==5.10.1",
        "pre-commit==3.6.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
