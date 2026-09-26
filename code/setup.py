from setuptools import setup, find_packages

setup(
    name='prism-aad',
    version='0.1.0',
    description='PRISM instance selection and feedback for automatic algorithm design',
    packages=find_packages(),
    package_dir={'': '.'},
    python_requires='>=3.9,<3.13',
    install_requires=[
        'numpy<2',
        'numba',
        'openai',
        'pytz',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'elkai',
    ]
)
