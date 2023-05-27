from setuptools import setup, find_packages

setup(
    name='flowchain',
    packages=find_packages(exclude=[]),
    version='0.0.7',
    license='MIT',
    description='Flowchain - Method Chaining for TensorFlow',
    author='OrigamiDream',
    author_email='hello@origamidream.me',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/OrigamiDream/flowchain',
    python_requires='>=3.7.0',
    extra_require={
        'gpu': ['tensorflow>=2.4'],
        'cpu': ['tensorflow-cpu>=2.4']
    },
    install_requires=[],
    keywords=[
        'machine learning',
        'deep learning',
        'tensorflow',
        'method chaining',
        'extension'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7'
    ]
)
