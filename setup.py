from setuptools import setup, find_packages

setup(
    name='climada_gambia',
    version='0.1',
    description='CLIMADA risk analysis for The Gambia',
    # url='http://github.com/ChrisFairless/climada_gambia',
    author='Chris Fairless',
    author_email='chrisfairless@hotmail.com',
    license='OSI Approved :: GNU Lesser General Public License v3 (GPLv3)',
    python_requires=">=3.11,<3.12",
    install_requires=[
        'climada',
        'climada_petals',
	    'contextily'
    ],
    packages=find_packages(),
    include_package_data=False
)
