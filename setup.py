#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    # will make this explicit in readme
    #'numpy>=1.9.2',
    #'pandas>=0.16.2',
    #'progressbar2>=3.3.0'
]

test_requirements = [
    'numpy>=1.9.2',
    'pandas>=0.16.2',
    'progressbar2>=3.3.0'
]

setup(
    name='pyrecommend',
    version='0.1.0',
    description="Simple user-item recommendations in python.",
    long_description=readme + '\n\n' + history,
    author="Andrés Bucchi",
    author_email='afbucchi@gmail.com',
    url='https://github.com/misterte/pyrecommend',
    packages=[
        'pyrecommend',
    ],
    package_dir={'pyrecommend':
                 'pyrecommend'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='pyrecommend',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.3',
        #'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
