from setuptools import setup, find_packages

setup(
    name='drift_control',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'category_encoders',
    ],
)
