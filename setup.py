from setuptools import find_packages, setup
from typing import List

HPY='-e .'

def get_reqirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HPY in requirements:
            requirements.remove(HPY)

    return requirements


setup(
    name='Zomatorpoject',
    version='0.0.1',
    author='akshat',
    author_email='chourasiaakshat2@gmail.com',
    install_requires = get_reqirements('requirements.txt'),
    packages=find_packages()
)