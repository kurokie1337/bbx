# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""BBX - Blackbox Workflow Engine"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bbx-workflow",
    version="1.0.0",
    author="Ilya Makarov",
    author_email="",
    description="Production-grade workflow automation platform with MCP integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kurokie1337/bbx",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core - Required
        "click>=8.1.7",
        "pyyaml>=6.0.1",
        "httpx>=0.25.2",
        "pydantic>=2.5.2",
        "jinja2>=3.1.2",
    ],
    extras_require={
        "server": [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
            "websockets>=12.0",
            "python-multipart>=0.0.6",
        ],
        "docker": [
            "docker>=6.1.3",
        ],
        "aws": [
            "boto3>=1.34.0",
        ],
        "browser": [
            "playwright>=1.40.0",
        ],
        "ai": [
            "llama-cpp-python>=0.2.90",
            "tqdm>=4.66.1",
        ],
        "mcp": [
            "mcp>=1.0.0",
        ],
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "ruff>=0.1.6",
            "mypy>=1.7.1",
        ],
        "full": [
            # All optional dependencies
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
            "websockets>=12.0",
            "python-multipart>=0.0.6",
            "docker>=6.1.3",
            "boto3>=1.34.0",
            "llama-cpp-python>=0.2.90",
            "tqdm>=4.66.1",
            "mcp>=1.0.0",
            "prometheus-client>=0.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bbx=cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "blackbox": ["py.typed"],
    },
    zip_safe=False,
    keywords=[
        "workflow",
        "automation",
        "orchestration",
        "devops",
        "cicd",
        "yaml",
        "dag",
        "mcp",
        "ai",
    ],
    project_urls={
        "Bug Reports": "https://github.com/kurokie1337/bbx/issues",
        "Source": "https://github.com/kurokie1337/bbx",
        "Documentation": "https://github.com/kurokie1337/bbx/tree/main/docs",
    },
)
