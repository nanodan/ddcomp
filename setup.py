from setuptools import setup, find_packages

setup(name='ddcomp',
    version='0.5',
    description='Delay discounting calculator',
    url='https://github.com/nanodan/ddcomp',
    author='Daniel J. Lewis',
    author_email='daniel.jacob.lewis@gmail.com',
    license='GNU General Public License v3.0',
    packages=find_packages(),
    install_requires=['numpy','scikit-learn','pandas','matplotlib'],
    package_data={'':['*.txt']},
    zip_safe=False)