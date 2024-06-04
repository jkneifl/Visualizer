# inspector3D
Tool for three-dimensional animations of geometries like FE meshes.  

![gui](doc/gui.png)


# Features
This repository implements a OpenGL-based gui for three-dimensional animations of geometries like FE meshes. 
It can be used to visualize results, e.g. from transient simulations or to inspect the geometry itself.
Moreover, it can be used to export the animations for presentations or publications.

## Installation

You can either clone the repository and install the package locally or install it directly from PyPI.

### PyPI

```bash
pip install visualizer-3d
```

### Local
Clone this repository and install it to your local environment as package using pip:

```bash
git clone git@github.com:jkneifl/visualizer.git
cd cd visualizer
pip install -e .
```

## Usage

The base class `Visualizer` can be used to create animations of geometries.
It has a method `animate` that can be used to create animations of geometries.

```python
from visualizer import Visualizer

coordinates = ...

# create a visualizer object
visualizer = Visualizer()
visualizer.animate(
    coordinates=coordinates,
)
```
It can animate point clouds, or meshes if the corresponding faces are provided.
For a detailed description of the parameters, see the docstring of the `animate` method.
An example of a simple animation is shown below.

![arm](doc/arm_rotating.gif)