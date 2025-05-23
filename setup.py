from setuptools import setup, find_packages

setup(
    name="machine-learning-deep-learning-from-scratch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
    ],
    author="hachemboudoukha",
    author_email="hachem.boudoukha@gmail.com",
    description="Implementation of machine learning and deep learning models from scratch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hachemboudoukha/machine-learning-deep-learning-from-scratch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 