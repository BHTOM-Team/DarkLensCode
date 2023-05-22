from setuptools import setup

requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='darklens',
    version='1.0.0',
    install_requires=requirements,
    entry_points = {
        'console_scripts': ['darklens=dark_lens_code:main', 'darklens-plot=plotting:main']
    }
)