from setuptools import find_packages , setup
from typing import List
hypen_e_dot='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This Function will return list of requirements
    
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)

    return requirements



setup(
name='MLproject',
version='0.0.1',
author='Harsh',
author_email='mailtopandeyharsh@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')


)