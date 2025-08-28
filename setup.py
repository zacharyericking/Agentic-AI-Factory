from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentic-ai-factory",
    version="0.1.0",
    author="Zachary King",
    description="A Python package that simplifies the generation of Agents using LangGraph with persona-based decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langgraph>=0.0.40",
        "langchain>=0.1.0",
        "langchain-google-genai>=1.0.0",
        "google-generativeai>=0.3.0",
        "pydantic>=2.0.0",
        "networkx>=3.0",
        "numpy>=1.21.0",
        "chromadb>=0.4.0",
        "neo4j>=5.0.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
)
