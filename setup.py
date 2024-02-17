from setuptools import find_packages, setup

setup(
    name="rosettas_repo",  # Replace with your project's name
    version="0.1.0",  # Your project's version
    author="Giovanni Campa",  # Your name or your organization's name
    author_email="giocampa93@gmail.com",  # Your contact email
    description="Implementation of various ML algorithms with pytorch and keras",  # A brief description of the project
    long_description=open(
        "readme.md"
    ).read(),  # A long description from your README file
    long_description_content_type="text/markdown",  # Specifies that the long description is in Markdown
    url="http://github.com/yourusername/your_project_name",  # Project home page or repository URL
    packages=find_packages(),  # Automatically find and include all packages in your project
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.0",
        "tensorflow==2.15",  # TensorFlow version with GPU support, assuming compatible CUDA is installed
        "scikit-learn==1.4.0",
        "torch==2.2.0",  # Example for PyTorch with CUDA 11.1 support
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
        "Programming Language :: Python :: 3",  # Specify the Python versions you support here
        "License :: OSI Approved :: MIT License",  # Your project's license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Minimum version requirement of Python
)
