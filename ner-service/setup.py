from setuptools import setup, find_packages
import os
import re

#__file__=os.path.abspath('.')

def find_version():
    fn = os.path.join(os.path.dirname(__file__), 'nermodel', 'version.py')
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              open(fn).read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='ner_serving',
    version=find_version(),
    description=
    'A simple ner service http service by aiohttp and torch. Support GPU.',
    packages=find_packages(),
    install_requires=[
        'torch<=0.4.1', 'msgpack', 'aiohttp', 'docopt','numpy',
    ],
    entry_points={
        'console_scripts': ['ner-serving=nermodel.cmd:entry_point'],
    })
