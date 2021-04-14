from setuptools import setup

setup(
    name='SimCLR-Keras',
    version='2.0',
    description='SimCLR Contrastive learning library for pretraining image processing models',
    author='Michiel Dhont and Laouen Belloli',
    package=['SimCLR-Keras', ''],
    scripts=['scripts/simCLR_pretrain_vgg16.py'],
    install_requires=[
        'tensorflow-gpu==2.2.0',
        'tensorboard',
        'pandas>=1.1',
        'scikit-learn',
        'scipy',
        'numpy',
        'DateTime',
    ]
)
