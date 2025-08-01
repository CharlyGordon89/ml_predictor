from setuptools import setup, find_packages

setup(
    name="ml_predictor",
    version="0.1.0",
    description="Reusable prediction module for loading trained ML models and making predictions",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "joblib",
        "pandas",
        "scikit-learn"  # or any other framework your model uses
    ],
    include_package_data=True,
    python_requires=">=3.7",
)
