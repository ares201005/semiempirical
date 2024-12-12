from os import path
import torch
import time
import seqm
from seqm.seqm_functions.constants import Constants, ev_kcalpmol
from seqm.basics import  Parser, Hamiltonian, Pack_Parameters, Energy
from seqm.seqm_functions.parameters import params

here = '/Applications/anaconda3/lib/python3.8/site-packages/PYSEQM'
torch.set_default_dtype(torch.float64)

device = torch.device('cpu')

species = torch.as_tensor([[6,8,1,1]],dtype=torch.int64, device=device)
coordinates = torch.tensor([
                  [
					[ 0.0000000000,  0.0000000000,  0.0000000000],
					[ 1.2195000000,  0.0000000000,  0.0000000000],
					[-0.5422000000,  0.9391179479,  0.0000000000],
					[-0.5422000000, -0.9391179479, -0.0000000000]
                  ]
                 ], device=device)

coordinates.requires_grad_(True)

elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM3
                   'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [2,0.0], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   #'sp2' : [True, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                   'sp2' : [False],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   #'learned' : ['U_ss'], # learned parameters name list, e.g ['U_ss']
                   'learned' : [], # learned parameters name list, e.g ['U_ss']
                   'parameter_file_dir' : here+'/seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   }

const = Constants().to(device)

seqm_parameters['eig']=True
with torch.autograd.set_detect_anomaly(True):
    eng = Energy(seqm_parameters).to(device)
    start = time.perf_counter()
    Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, coordinates, species, learned_parameters=dict(), all_terms=True)
    finish = time.perf_counter() # for energy
    L=Etot.sum()
    #L.backward()
    #"""
    coordinates.grad=torch.autograd.grad(L,coordinates,
                                         #grad_outputs=torch.tensor([1.0]),
                                         create_graph=True,
                                         retain_graph=True)[0]
    # finish = time.perf_counter() # for gradient
    #"""
print("Timing: ", finish-start , " (s)")
#print("Orbital Energy (eV): ", e.tolist())
print("Electronic Energy (eV): ", Eelec.tolist())
print("Nuclear Energy (eV): ", Enuc.tolist())
print("Total Energy (eV): ", Etot.tolist())
#print("Heat of Formation (kcal/mol): ", (Hf*ev_kcalpmol).tolist())
'''
#print(coordinates.grad)
#print(p.grad)
#"""
if const.do_timing:
    print(const.timing)

coordinates = torch.tensor([
                  [
                   [  0.0000000,    0.0000000,    0.0000000], 
                   [  0.0000000,    0.0000000,   12.0000000], 
                  ]
                 ], device=device)

print('---')
with torch.autograd.set_detect_anomaly(True):
    Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, coordinates, species, learned_parameters=dict(), all_terms=True)
print("Electronic Energy (eV): ", Eelec.tolist())
print("Nuclear Energy (eV): ", Enuc.tolist())
print("Total Energy (eV): ", Etot.tolist())
'''
