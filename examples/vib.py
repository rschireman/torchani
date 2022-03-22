# -*- coding: utf-8 -*-
"""
Computing Vibrational Frequencies Using Analytical Hessian
==========================================================

TorchANI is able to use ASE interface to do structure optimization and
vibration analysis, but the Hessian in ASE's vibration analysis is computed
numerically, which is slow and less accurate.

TorchANI therefore provide an interface to compute the Hessian matrix and do
vibration analysis analytically, thanks to the super power of `torch.autograd`.
"""
import ase
import ase.optimize
import ase.vibrations
from numpy import int32
import torch
import torchani
import math
import os
import torchani.ase
import torchani.data
from torchani.utils import ChemicalSymbolsToInts
###############################################################################
# Let's now manually specify the device we want TorchANI to run:
device = torch.device('cpu')
Rcr = 5.2000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
species_order = ['H','C','S']

num_species = len(species_order)
aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
energy_shifter = torchani.utils.EnergyShifter(None)
###############################################################################
# The code to define networks, optimizers, are mostly the same
aev_dim = aev_computer.aev_length

H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

S_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)


nn = torchani.ANIModel([H_network, C_network, S_network])
best_pt = torch.load('force-training-best.pt')
nn.load_state_dict(best_pt)
model = torchani.nn.Sequential(aev_computer, nn).to(device)

molecule = ase.io.read("btbt_0_1_2.pdb")

ase_calc = torchani.ase.Calculator(model=model, species=species_order)
molecule.calc = ase_calc

opt = ase.optimize.BFGS(molecule)
opt.run()
ase.io.write('opt.pdb',molecule)

vib = ase.vibrations.Vibrations(molecule)
vib.run()
vib.get_vibrations(read_cache=False)
vib.summary()
vib.write_jmol()
###############################################################################
# Now let's extract coordinates and species from ASE to use it directly with
# TorchANI:
species = torch.tensor(molecule.get_atomic_numbers(), device=device, dtype=torch.long).unsqueeze(0)
# species = molecule.get_chemical_symbols()
coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).requires_grad_(True).float()

# species_to_tensor = ChemicalSymbolsToInts(species)

# index_tensor = species_to_tensor(species_order)
# print(index_tensor)

###############################################################################
# TorchANI needs the masses of elements in AMU to compute vibrations. The
# masses in AMU can be obtained from a tensor with atomic numbers by using
# this utility:

masses = torchani.utils.get_atomic_masses(species)
print(masses)
###############################################################################
# To do vibration analysis, we first need to generate a graph that computes
# energies from species and coordinates. The code to generate a graph of energy
# is the same as the code to compute energy:
cell = torch.tensor(molecule.get_cell()).float()
pbc = torch.tensor([True,True,True])

# species = torch.tensor([[2,	1,	1,	0,	1,	0,	1,	0,	1,	0,	1,	1,	2,	1,	1,	0,	1,	0,	1,	0,	1,	1,	1,	0]],device=device,dtype=torch.long)

species = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	2,	2,	2,	2,
]],device=device,dtype=torch.long)

energies = model((species, coordinates),cell=cell, pbc=pbc).energies
# energies = model((species, coordinates)).energies
###############################################################################
# We can now use the energy graph to compute analytical Hessian matrix:
hessian = torchani.utils.hessian(coordinates,energies=energies)

###############################################################################
# The Hessian matrix should have shape `(1, 9, 9)`, where 1 means there is only
# one molecule to compute, 9 means `3 atoms * 3D space = 9 degree of freedom`.
print(hessian.shape)

###############################################################################
# We are now ready to compute vibrational frequencies. The output has unit
# cm^-1. Since there are in total 9 degree of freedom, there are in total 9
# frequencies. Only the frequencies of the 3 vibrational modes are interesting.
# We output the modes as MDU (mass deweighted unnormalized), to compare with ASE.
freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDU')
torch.set_printoptions(precision=3, sci_mode=False)

print('Frequencies (cm^-1):', freq)
print('Force Constants (mDyne/A):', fconstants)
print('Reduced masses (AMU):', rmasses)
print('Modes:', modes)