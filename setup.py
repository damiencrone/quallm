from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='quallm',
    version='0.8.2',
    author='Damien Crone',
    author_email='damien.crone@gmail.com',
    description='A Python library for LLM-assisted content analysis tasks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/damiencrone/quallm',
    packages=find_packages(),
    package_data={
        'quallm': ['data/diagnostics/*.json'],
    },
    include_package_data=True,
    install_requires=[
        'pydantic',
        'openai',
        'instructor',
        'futures',
        'pandas',
        'numpy',
        'litellm',
        'ollama',
        'umap-learn',
        'hdbscan'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.9',
)