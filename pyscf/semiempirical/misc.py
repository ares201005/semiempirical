import os
import numpy

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        from pyscf import __path__ as ext_modules
        for path in ext_modules:
            libpath = os.path.join(path, 'lib')
            if os.path.isdir(libpath):
                for files in os.listdir(libpath):
                    if files.startswith(libname):
                        return numpy.ctypeslib.load_library(libname, libpath)
        raise