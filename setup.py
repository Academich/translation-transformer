from setuptools import setup

if __name__ == '__main__':
    setup(
        name='molecular_transformer_lightning',
        version='0.1.0',
        description='SMILES-to-SMILES Transformer for chemical reactions',
        author='Mikhail Andronov',
        packages=['molecular_transformer_lightning'],
        install_requires=[
            'lightning',
            'jsonargparse[signatures]',
            'rdkit',
            'tensorboard',
            'gdown',
        ],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.10',
        ],
    )
