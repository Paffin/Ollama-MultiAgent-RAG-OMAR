from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="omar",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MultiAgent System with RAG and Analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/omar",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.32.0",
        "plotly>=5.18.0",
        "pandas>=2.2.0",
        "requests>=2.31.0",
        "nest-asyncio>=1.6.0",
        "python-dotenv>=1.0.0",
        "typing-extensions>=4.9.0",
    ],
    entry_points={
        "console_scripts": [
            "omar=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "omar": ["config/*.json", "prompts/*.txt"],
    },
) 