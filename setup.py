"""
Setup script for Seawaste Detection Model
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='seawaste-detection',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Custom deep learning model for detecting marine debris and waste',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Aztigma-Thesis/Object-detection-Thesis',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'seawaste-train=scripts.train:main',
            'seawaste-detect=scripts.detect:main',
            'seawaste-eval=scripts.evaluate:main',
        ],
    },
    keywords='object-detection, marine-debris, seawaste, yolo, computer-vision, deep-learning',
    project_urls={
        'Bug Reports': 'https://github.com/Aztigma-Thesis/Object-detection-Thesis/issues',
        'Source': 'https://github.com/Aztigma-Thesis/Object-detection-Thesis',
    },
)
