import os
import setuptools

merfishdecoder_version = '0.1.0'

install_requires = [line.rstrip() for line in open(
    os.path.join(os.path.dirname(__file__), "requirements.txt"))]

setuptools.setup(
    name='merfishdecoder',
    version=merfishdecoder_version,
    description="MERFISH Decoder",
    author='Rongxin Fang',
    author_email='r3fang@fas.harvard.edu',
    packages=setuptools.find_packages(),
    license='LICENSE.txt',
    install_requires=install_requires,
    keywords = ["Bioinformatics pipeline",
                "MERFISH",
                "Multiplexed FISH",
                "Genome-wide imaging"],
    scripts = ["bin/merfishdecoder"]
)
