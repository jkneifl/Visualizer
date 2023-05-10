from setuptools import find_packages, setup

setup(
    name='inspector3D',
    packages=find_packages(include=['inspector3D', 'inspector3D.*']),
    version='1.0',
    description='Tool for three-dimensional animations of geometries like FE meshes',
    author='Jonas Kneifl',
    license='MIT',
    install_requires=['opencv-python',
                      'pyqtgraph',
                      'PyOpenGL',
                      'PyQt6',
                      'Pillow',
                      'matplotlib'],
    setup_requires=[],
    tests_require=[],
    test_suite='tests',
)
