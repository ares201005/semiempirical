from .mindo3 import RMINDO3, UMINDO3
from .mndo_class import RMNDO
from .omx_class import ROM2

__version__ = '0.1.0'

def MINDO3(mol):
    if mol.spin == 0:
        return RMINDO3(mol)
    else:
        return UMINDO3(mol)

def MNDO(mol, model):
    if mol.spin == 0:
        return RMNDO(mol, model)

def OMX(mol, model):
    if mol.spin == 0:
        return ROM2(mol, model)
