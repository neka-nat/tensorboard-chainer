#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    'protobuf',
    'six',
    'moviepy'
]

test_requirements = [
    'pytest',
    'moviepy'
]

setup(
    name='tensorboard-chainer',
    version='0.5.2',
    description='Log TensorBoard events with chainer',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='nake nat',
    author_email='nakanat.stock@gmail.com',
    url='https://github.com/neka-nat/tensorboard-chainer',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    test_suite='tests',
    tests_require=test_requirements
)

# python setup.py bdist_wheel --universal upload
