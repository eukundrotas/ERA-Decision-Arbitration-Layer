from setuptools import setup, find_packages

setup(
    name="era-dal",
    version="1.0.0",
    description="ERA Decision & Arbitration Layer - Reliable LLM Ensemble Decision Making",
    author="ERA Team",
    author_email="support@era-dal.example.com",
    url="https://github.com/yourusername/era-dal",
    packages=find_packages(),
    install_requires=[
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
        "requests==2.31.0",
        "jsonschema==4.20.0",
        "pandas==2.1.0",
        "openpyxl==3.1.0",
        "pyyaml==6.0.1",
        "scipy==1.11.0",
        "click==8.1.7",
        "colorama==0.4.6",
    ],
    entry_points={
        "console_scripts": [
            "era-dal=app:main",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
