from setuptools import setup, find_packages

setup(name="statfenics",
      version="0.0.1",
      author="Connor Duffin",
      author_email="connor.p.duffin@gmail.com",
      description="Tools for statFEM problems with Fenics",
      license="MIT",
      packages=find_packages(),
      install_requires=[
          "numpy", "scipy", "fenics", "petsc4py", "pytest"
      ],
      classifiers=(
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ))
