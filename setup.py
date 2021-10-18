from setuptools import setup, find_packages

VERSION = '0.0.1'

DESCRIPTION = 'My personal training ANI engine for TORCHANI'

LONG_DESCRIPTION = "Based on ANIEngine, but more personal to the changes I need to make in my work. I also strictly used .py files that I wrote, which are personally easier for me to understand and modify."

setup(
        name = "personal_trainer", 
        version = VERSION, 
        author = "Kate Huddleston", 
        author_email = "kdavis2@ufl.edu", 
        description = DESCRIPTION, 
        long_description = LONG_DESCRIPTION, 
        packages = find_packages(), 
        install_requires=[], 

        keywords=['python', 'torchani', 'torch'], 
        classifiers = [
            "Intended AUdience :: Beginner ANI users", 
            "Programming Language :: Python :: 3", 
            "Operating System :: Linux"
            ]
        )



