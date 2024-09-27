import io
import re
import setuptools


with open('mgktools/__init__.py') as fd:
    __version__ = re.search("__version__ = '(.*)'", fd.read()).group(1)


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')

setuptools.setup(
    name='mgktools',
    version=__version__,
    python_requires='>=3.10',
    install_requires=[
        'scikit-learn>=0.24.1',
        'tqdm>=4.62.0',
        'hyperopt>=0.2.5',
        'optuna>=3.6.0',
        'scipy>=1.6.2',
        'mendeleev>=0.7',
        'rxntools>=0.0.2',
        'pycuda>=2022.1',
        'rdkit>=2022.9.2',
        'deepchem==2.7.2.dev20231207083329',
        'typed-argument-parser',
        'ipython',
    ],
    entry_points={
        'console_scripts': [
            'mgk_read_data=mgktools.exe.run:mgk_read_data',
            'mgk_kernel_calc=mgktools.exe.run:mgk_kernel_calc',
            'mgk_model_evaluate=mgktools.exe.run:mgk_model_evaluate',
            'mgk_embedding=mgktools.exe.run:mgk_embedding',
            'mgk_hyperopt=mgktools.exe.run:mgk_hyperopt',
            'mgk_hyperopt_multi_datasets=mgktools.exe.run:mgk_hyperopt_multi_datasets',
            'mgk_optuna=mgktools.exe.run:mgk_optuna',
            'mgk_optuna_multi_datasets=mgktools.exe.run:mgk_optuna_multi_datasets',
        ]
    },
    author='Yan Xiang',
    author_email='1993.xiangyan@gmail.com',
    description='Marginalized graph kernel library for molecular property prediction',
    long_description=long_description,
    url='https://github.com/xiangyan93/mgktools',
    packages=setuptools.find_namespace_packages(include=['mgktools', 'mgktools.*']),
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    package_data={'': ['hyperparameters/configs/*.json']}
)
