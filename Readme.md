ANNonGPU
========

This library implements the [algorithm of Carleo and Troyer](https://arxiv.org/abs/1606.02318) for respresenting quantum many-body wavefunction using various artificial neural network (ANN) architectures on the GPU.

Installation
============

```
cd ANNonGPU
cmake . -DCMAKE_INSTALL_PREFIX=<e.g. $HOME/.local>
make install

(for python interface only)
cd pyANNonGPU
python setup.py install
```
