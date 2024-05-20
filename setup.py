from setuptools import setup, find_packages

setup(
    name="state_abstraction_rl",
    version="0.0.1",
    install_requires=[
        "gymnasium>=0.26.2",
        "pygame==2.1.3.dev8",
        "imageio>=2.24.0",
        "matplotlib>=3.6.3",
        "pandas",
        "seaborn",
        "torch",
        "hippocluster @ git+https://github.com/echalmers/hippocluster",
        "distinctipy"
    ],
    packages=find_packages(),
)
