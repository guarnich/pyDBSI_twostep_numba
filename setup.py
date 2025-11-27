from setuptools import setup, find_packages

setup(
    name="dbsi_toolbox_numba",
    version="0.3.0", 
    author="Francesco Guarnaccia",
    description="A comprehensive toolbox for Diffusion Basis Spectrum Imaging (DBSI) with Numba acceleration.",
    url="https://github.com/guarnich/pyDBSI",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'dbsi-fit=scripts.dbsi_cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "nibabel>=3.2.0",
        "dipy>=1.5.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "torch>=1.12.0",
        "numba>=0.55.0",  
    ],
)