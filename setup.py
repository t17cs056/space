from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include #cimport numpy を使うため

ext = Extension("synthetic", sources=["synthetic.pyx"], include_dirs=['.', get_include()]) #ファイルごとに""内を変える
setup(name="synthetic", ext_modules=cythonize([ext]))

#コンパイルコマンド
# > python setup.py build_ext --inplace