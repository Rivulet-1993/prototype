import setuptools


setuptools.setup(
    name='Spring-Prototype',
    version='1.0.0',
    author='Yuan Kun',
    author_email='yuankun@sensetime.com',
    description='Distributed Framework for Image Classification',
    url='http://gitlab.bj.sensetime.com/project-spring/prototype.git',
    packages=setuptools.find_packages(),
    license='Internal',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: Internal',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ]
)
