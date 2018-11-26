from setuptools import setup

NAME = 'phik_python'


def setup_package() -> None:
    """
    The main setup method. It is responsible for setting up and installing the package.
    """
    setup(name=NAME,
          url='http://phik.kave.io',
          license='',
          author='KPMG',
          author_email='phik@phik',
          description='PhiK test package',
          python_requires='>=3.5',
          packages=['phik_python'],
          install_requires=[]
          )


if __name__ == '__main__':
    setup_package()
