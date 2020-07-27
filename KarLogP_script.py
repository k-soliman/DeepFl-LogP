#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:59:00 2020

@author: Kareem Soliman
"""

import numpy as np 
import pandas as pd
from rdkit import Chem
import tensorflow as tf
from tensorflow import keras
#from keras.models import load_model

model = tf.keras.models.load_model(r"2020_DNN_v44_epoch_78_r0.892_rms0.359.h5")

mol_smile = input("Enter molecule: ")

mol = Chem.MolFromSmiles(mol_smile)


"""**Functions**"""

from rdkit import rdBase
from rdkit import RDConfig
from rdkit.Chem import Descriptors
from rdkit import DataStructs

def myfunction():
    value = mol.GetNumAtoms()
    return value


def fr_Trithiolane(mol):
  patt = Chem.MolFromSmarts('[#16]-[#6]-1-[#6]-[#16]-[#16]-[#16]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_Dithiolane(mol):
  patt = Chem.MolFromSmarts('[#16]-[#6]-1-[#6]-[#16]-[#16]-[#16]-1')
  x = mol.HasSubstructMatch(patt)                               
  x = int(x == True)
  return x 

def fr_DiSulf(mol):
  patt = Chem.MolFromSmarts('[#16](-[#16]-[*])-[*]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_Chlorobenzene(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[Cl]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_dichlorobenzene12(mol):
  patt = Chem.MolFromSmarts('c1ccc(c(c1)-[Cl])-[Cl]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x


def fr_dichlorobenzene14(mol):
  patt = Chem.MolFromSmarts('c1cc(ccc1-[Cl])-[Cl]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_C4(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#6]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x


def fr_C8(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x


def fr_propanoic_SMART(mol):
  patt = Chem.MolFromSmarts('[#8]-[#6]-[#6]-[#6](=[#8])-[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_propanoic(mol):
  patt = Chem.MolFromSmarts('[#8]-[#6]-[#6]-[#6](=[#8])-[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_urea(mol):
  patt = Chem.MolFromSmarts('[#6](-[#7])(=[#8])-[#7]-[#6]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_so2(mol):
  patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])-[#8-]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_quinoline(mol):
  patt = Chem.MolFromSmarts('c1ccc2c(c1)cccn2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_cyclpropnitro(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#7]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_Thiadiazole(mol):
  patt = Chem.MolFromSmarts('c1nncs1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_dioxolane(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#8]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_Pyrazinedihydro(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#7]=[#6]-[#6]=[#7]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_pyridine(mol):
  patt = Chem.MolFromSmarts('c1ccncc1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_betalactone(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-1=[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_nitrosomorpholine(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#6]-[#7]-1-[#7]=[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_dichlorodioxane(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6](-[#6](-[#8]-1)-[Cl])-[Cl]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_dioxane(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#6]-[#8]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_metTHF(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6]-1-[#6]-[#6]-[#6]-[#8]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 

def fr_4pyranone(mol):
  patt = Chem.MolFromSmarts('[#6]-1=[#6]-[#8]-[#6]=[#6]-[#6]-1=[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x 
  
  
def fr_THF(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]-[#8]-[#6]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Dimethoxyethane(mol):
  patt = Chem.MolFromSmarts('[#6]-[#8]-[#6]-[#6]-[#8]-[#6]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Dihydropyran(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]=[#6]-[#8]-[#6]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Tetrahydropyran(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]-[#8]-[#6]-[#6]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Isoxazole(mol):
  patt = Chem.MolFromSmarts('c1conc1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Thiazole(mol):
  patt = Chem.MolFromSmarts('c1cscn1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Pyridazine(mol):
  patt = Chem.MolFromSmarts('c1ccnnc1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Pyrimidine(mol):
  patt = Chem.MolFromSmarts('c1cncnc1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Pyrazine(mol):
  patt = Chem.MolFromSmarts('c1cnccn1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Dimethylisoxazole(mol):
  patt = Chem.MolFromSmarts('[#6]-c1cc(no1)-[#6]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Oxetane(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Arginine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](-[#6](=[#8])-[#7])-[#7])-[#6]-[#7]=[#6](-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Proline(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6](-[#7]-[#6]-1)-[#6](=[#8])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Tryptophane(mol):
  patt = Chem.MolFromSmarts('c1ccc2c(c1)c(cn2)-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Alanine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Lysine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6]-[#6]-[#7])-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Phenylalanine(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Tyrosine(mol):
  patt = Chem.MolFromSmarts('c1cc(ccc1-[#6]-[#6](-[#6](=[#8])-[#7])-[#7])-[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Methionine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#16]-[#6]-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Leucine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6](-[#6])-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Isoleucine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6]-[#6](-[#6])-[#6](-[#6](=[#8])-[#8])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Valine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6](-[#6])-[#6](-[#6](=[#8])-[#8])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Glutamate(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](=[#8])-[#8])-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Glutamine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](=[#8])-[#7])-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Aspartate(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#8])-[#7])-[#6](=[#8])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Glycine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Histidine(mol):
  patt = Chem.MolFromSmarts('c1c(ncn1)-[#6]-[#6](-[#6](=[#8])-[#8])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Serine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Threonine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6](-[#6](-[#6](=[#8])-[#8])-[#7])-[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Asparagine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#6](=[#8])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Cysteine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#16]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Chromene(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]=[#6]-c1ccccc1-[#8]-2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Chromene_2(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#6]-c1ccccc1-[#8]-2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Chromane(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]-c1ccccc1-[#8]-[#6]-2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Chromanone(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#8]-c1ccccc1-[#6]-2=[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Chromone_2(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#8]-c1ccccc1-[#6]-2=[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Furan_2(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]=[#6]-[#6]-[#8]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Oxazoline(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]=[#7]-1')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Nitrobenzene(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[#7+](=[#8])-[#8-]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Thiophene_N(mol):
  patt = Chem.MolFromSmarts('[#7]-2-c1ccccc1-[#16]-[#6]-2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Quinolonium(mol):
  patt = Chem.MolFromSmarts('[#6]-c3cc[n+]c4ccccc34')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Benzimidazole(mol):
  patt = Chem.MolFromSmarts('c1ccc2c(c1)ncn2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Chlorzoxazone(mol):
  patt = Chem.MolFromSmarts('c1cc-2c(cc1-[Cl])-[#7]-[#6](=[#8])-[#8]-2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Naphthalene(mol):
  patt = Chem.MolFromSmarts('c2ccc1ccccc1c2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Hbond(mol):
  patt = Chem.MolFromSmarts('[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_quatNwC(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#7+](-[#6])(-[#6])-[#6]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Ammoniopropyl(mol):
  patt = Chem.MolFromSmarts('[#6]-[#7+](-[#6])-[#6]-[#6]-[#6]-[#7+]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Aniline(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_ACA(mol):
  patt = Chem.MolFromSmarts('[*]-[#6]-[*]-[#6]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_ANH(mol):
  patt = Chem.MolFromSmarts('[*]-[#7]-[#1]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_AN(mol):
  patt = Chem.MolFromSmarts('[*]-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_COdbO(mol):
  patt = Chem.MolFromSmarts('[#6]=[#6]=[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_CdbNdbO(mol):
  patt = Chem.MolFromSmarts('[#6]=[#7+]=[#8-]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_CdbN(mol):
  patt = Chem.MolFromSmarts('[#6]=[#7+]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_CN(mol):
  patt = Chem.MolFromSmarts('[#6]-[#7+]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_NSO(mol):
  patt = Chem.MolFromSmarts('[#7]=[#16]=[#8]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_NN(mol):
  patt = Chem.MolFromSmarts('[#7](=[#7+]=[*])-[*]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Etsub(mol):
  patt = Chem.MolFromSmarts('c1cc[n+](cc1)-[*]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_ringNO(mol): 
  patt = Chem.MolFromSmarts('[n+]1(c(oc2ccccc12)-[*])-[*]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Etplus(mol): 
  patt = Chem.MolFromSmarts('[n+]3(c1cc(ccc1c2ccc(cc2c3)-[#7])-[#7])-[*]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_benzoicsulfonic(mol):
  patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])(-[*])-[*]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Etroot(mol):
  patt = Chem.MolFromSmarts('[#6]-[#7+](-[#6])(-[#6]-[#6]-[#6]-[n+]1ccccc1)-[#6]-[#6]-[#6]-[#7+](-[#6])(-[#6])-[#6]-[#6]-[#6]-[n+]2ccccc2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Benzoxazolium(mol):
  patt = Chem.MolFromSmarts('[n+]1coc2ccccc12')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Benzoxazol(mol): 
  patt = Chem.MolFromSmarts('[n]1coc2ccccc12')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_Dapi(mol):
  patt = Chem.MolFromSmarts('c1cc(ccc1-[*])-[#6](=[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_HemiBabim(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[#6](=[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def fr_HemiBabim_2(mol):
  patt = Chem.MolFromSmarts('c1nc2c(n1)cccc2')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

# counting fragments

def ct_Trithiolane(mol):
  patt = Chem.MolFromSmarts('[#16]-[#6]-1-[#6]-[#16]-[#16]-[#16]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_Dithiolane(mol):
  patt = Chem.MolFromSmarts('[#16]-[#6]-1-[#6]-[#16]-[#16]-[#16]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_DiSulf(mol):
  patt = Chem.MolFromSmarts('[#16](-[#16]-[*])-[*]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_Chlorobenzene(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[Cl]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_dichlorobenzene12(mol):
  patt = Chem.MolFromSmarts('c1ccc(c(c1)-[Cl])-[Cl]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x


def ct_dichlorobenzene14(mol):
  patt = Chem.MolFromSmarts('c1cc(ccc1-[Cl])-[Cl]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x


def ct_propanoic(mol):
  patt = Chem.MolFromSmarts('[#8]-[#6]-[#6]-[#6](=[#8])-[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_urea(mol):
  patt = Chem.MolFromSmarts('[#6](-[#7])(=[#8])-[#7]-[#6]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x


# new ends here

def ct_so2(mol):
  patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])-[#8-]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_quinoline(mol):
  patt = Chem.MolFromSmarts('c1ccc2c(c1)cccn2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_cyclpropnitro(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#7]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_Thiadiazole(mol):
  patt = Chem.MolFromSmarts('c1nncs1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_dioxolane(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#8]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_Pyrazinedihydro(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#7]=[#6]-[#6]=[#7]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_pyridine(mol):
  patt = Chem.MolFromSmarts('c1ccncc1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_betalactone(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-1=[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_nitrosomorpholine(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#6]-[#7]-1-[#7]=[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_dichlorodioxane(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6](-[#6](-[#8]-1)-[Cl])-[Cl]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_dioxane(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#6]-[#8]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_metTHF(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6]-1-[#6]-[#6]-[#6]-[#8]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 

def ct_4pyranone(mol):
  patt = Chem.MolFromSmarts('[#6]-1=[#6]-[#8]-[#6]=[#6]-[#6]-1=[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x 
  
  
def ct_THF(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]-[#8]-[#6]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Dimethoxyethane(mol):
  patt = Chem.MolFromSmarts('[#6]-[#8]-[#6]-[#6]-[#8]-[#6]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Dihydropyran(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]=[#6]-[#8]-[#6]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Tetrahydropyran(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]-[#8]-[#6]-[#6]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Isoxazole(mol):
  patt = Chem.MolFromSmarts('c1conc1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Thiazole(mol):
  patt = Chem.MolFromSmarts('c1cscn1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Pyridazine(mol):
  patt = Chem.MolFromSmarts('c1ccnnc1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Pyrimidine(mol):
  patt = Chem.MolFromSmarts('c1cncnc1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Pyrazine(mol):
  patt = Chem.MolFromSmarts('c1cnccn1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Dimethylisoxazole(mol):
  patt = Chem.MolFromSmarts('[#6]-c1cc(no1)-[#6]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Oxetane(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Arginine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](-[#6](=[#8])-[#7])-[#7])-[#6]-[#7]=[#6](-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Proline(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6](-[#7]-[#6]-1)-[#6](=[#8])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Tryptophane(mol):
  patt = Chem.MolFromSmarts('c1ccc2c(c1)c(cn2)-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Alanine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Lysine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6]-[#6]-[#7])-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Phenylalanine(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Tyrosine(mol):
  patt = Chem.MolFromSmarts('c1cc(ccc1-[#6]-[#6](-[#6](=[#8])-[#7])-[#7])-[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Methionine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#16]-[#6]-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Leucine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6](-[#6])-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Isoleucine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6]-[#6](-[#6])-[#6](-[#6](=[#8])-[#8])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Valine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6](-[#6])-[#6](-[#6](=[#8])-[#8])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Glutamate(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](=[#8])-[#8])-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Glutamine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](=[#8])-[#7])-[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Aspartate(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#8])-[#7])-[#6](=[#8])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Glycine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](=[#8])-[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Histidine(mol):
  patt = Chem.MolFromSmarts('c1c(ncn1)-[#6]-[#6](-[#6](=[#8])-[#8])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Serine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Threonine(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6](-[#6](-[#6](=[#8])-[#8])-[#7])-[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Asparagine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#6](=[#8])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Cysteine(mol):
  patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#16]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Chromene(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]=[#6]-c1ccccc1-[#8]-2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Chromene_2(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#6]-c1ccccc1-[#8]-2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Chromane(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]-c1ccccc1-[#8]-[#6]-2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Chromanone(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#8]-c1ccccc1-[#6]-2=[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Chromone_2(mol):
  patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#8]-c1ccccc1-[#6]-2=[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Furan_2(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]=[#6]-[#6]-[#8]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Oxazoline(mol):
  patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]=[#7]-1')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Nitrobenzene(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[#7+](=[#8])-[#8-]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Thiophene_N(mol):
  patt = Chem.MolFromSmarts('[#7]-2-c1ccccc1-[#16]-[#6]-2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Quinolonium(mol):
  patt = Chem.MolFromSmarts('[#6]-c3cc[n+]c4ccccc34')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Benzimidazole(mol):
  patt = Chem.MolFromSmarts('c1ccc2c(c1)ncn2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Chlorzoxazone(mol):
  patt = Chem.MolFromSmarts('c1cc-2c(cc1-[Cl])-[#7]-[#6](=[#8])-[#8]-2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Naphthalene(mol):
  patt = Chem.MolFromSmarts('c2ccc1ccccc1c2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Hbond(mol):
  patt = Chem.MolFromSmarts('[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_quatNwC(mol):
  patt = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#7+](-[#6])(-[#6])-[#6]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Ammoniopropyl(mol):
  patt = Chem.MolFromSmarts('[#6]-[#7+](-[#6])-[#6]-[#6]-[#6]-[#7+]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Aniline(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_ACA(mol):
  patt = Chem.MolFromSmarts('[*]-[#6]-[*]-[#6]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_ANH(mol):
  patt = Chem.MolFromSmarts('[*]-[#7]-[#1]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_AN(mol):
  patt = Chem.MolFromSmarts('[*]-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_COdbO(mol):
  patt = Chem.MolFromSmarts('[#6]=[#6]=[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_CdbNdbO(mol):
  patt = Chem.MolFromSmarts('[#6]=[#7+]=[#8-]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_CdbN(mol):
  patt = Chem.MolFromSmarts('[#6]=[#7+]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_CN(mol):
  patt = Chem.MolFromSmarts('[#6]-[#7+]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_NSO(mol):
  patt = Chem.MolFromSmarts('[#7]=[#16]=[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_NN(mol):
  patt = Chem.MolFromSmarts('[#7](=[#7+]=[*])-[*]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Etsub(mol):
  patt = Chem.MolFromSmarts('c1cc[n+](cc1)-[*]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_ringNO(mol): 
  patt = Chem.MolFromSmarts('[n+]1(c(oc2ccccc12)-[*])-[*]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Etplus(mol): 
  patt = Chem.MolFromSmarts('[n+]3(c1cc(ccc1c2ccc(cc2c3)-[#7])-[#7])-[*]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_benzoicsulfonic(mol):
  patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])(-[*])-[*]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Etroot(mol):
  patt = Chem.MolFromSmarts('[#6]-[#7+](-[#6])(-[#6]-[#6]-[#6]-[n+]1ccccc1)-[#6]-[#6]-[#6]-[#7+](-[#6])(-[#6])-[#6]-[#6]-[#6]-[n+]2ccccc2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Benzoxazolium(mol):
  patt = Chem.MolFromSmarts('[n+]1coc2ccccc12')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Benzoxazol(mol):
  patt = Chem.MolFromSmarts('[n]1coc2ccccc12')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_Dapi(mol):
  patt = Chem.MolFromSmarts('c1cc(ccc1-[*])-[#6](=[#7])-[#7]')
  x = mol.HasSubstructMatch(patt)
  x = int(x == True)
  return x

def ct_HemiBabim(mol):
  patt = Chem.MolFromSmarts('c1ccc(cc1)-[#6](=[#7])-[#7]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_HemiBabim_2(mol):
  patt = Chem.MolFromSmarts('c1nc2c(n1)cccc2')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_phosphorous(mol):
  patt = Chem.MolFromSmarts('[P]-[#8]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_phosphorous_negative(mol):
  patt = Chem.MolFromSmarts('[P]-[#8-]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_so2neutral(mol):
  patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

def ct_fluorobenzene(mol):
  patt = Chem.MolFromSmarts('c1cc(c(c(c1-[F])-[F])-[F])-[F]')
  x = mol.GetSubstructMatches(patt)
  x = len(x)
  return x

"""Test"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import Descriptors

def fps_plus_others(mol):
  prints = Fingerprinter.FingerprintMol(mol)[0] 
  Ring = Descriptors.RingCount(mol)
  HeteroRing = Descriptors.NumAromaticHeterocycles(mol)
  Carbocycles = Descriptors.NumSaturatedCarbocycles(mol)
  nAromaticRing = Descriptors.NumAromaticRings(mol)
  nOHnNH = Descriptors.NHOHCount(mol)
  nNO = Descriptors.NOCount(mol)
  fr121 = fr_quatNwC(mol)
  fr122 = fr_Ammoniopropyl(mol)
  fr123 = fr_Aniline(mol)
  fr124 = fr_ACA(mol)
  fr125 = fr_AN(mol)
  fr126 = fr_ANH(mol)
  fr127 = fr_COdbO(mol)
  fr128 = fr_CdbNdbO(mol)
  fr129 = fr_CdbN(mol)
  fr130 = fr_CN(mol)
  fr131 = fr_Etsub(mol)
  fr132 = fr_ringNO(mol)
  fr133 = fr_Etplus(mol)
  fr134 = fr_benzoicsulfonic(mol)
  fr135 = fr_Etroot(mol)
  fr136 = fr_Benzoxazolium(mol)
  fr137 = fr_Benzoxazol(mol)
  fr138 = fr_Dapi(mol)
  fr139 = fr_HemiBabim(mol)
  fr140 = fr_HemiBabim_2(mol)
  fr141 = Descriptors.fr_benzene(mol)
  fr142 = Descriptors.fr_ether(mol)
  fr143 = Descriptors.fr_halogen(mol)
  fr144 = Descriptors.fr_Ndealkylation1(mol)
  fr145 = Descriptors.fr_Ndealkylation2(mol)
  fr146 = Descriptors.fr_aldehyde(mol)
  fr147 = Descriptors.fr_ketone(mol) 
  fr148 = mol.GetNumAtoms()
  fr0 = Descriptors.fr_bicyclic(mol)
  fr1 = Descriptors.fr_tetrazole(mol)
  fr2 = Descriptors.fr_oxime(mol)
  fr3 = Descriptors.fr_imidazole(mol)
  fr4 = Descriptors.fr_COO(mol)
  fr5 = Descriptors.fr_COO2(mol)
  fr6 = Descriptors.fr_C_O(mol)
  fr7 = Descriptors.fr_C_O_noCOO(mol)
  fr8 = Descriptors.fr_C_S(mol)
  fr9 = Descriptors.fr_HOCCN(mol)
  fr10 = Descriptors.fr_Imine(mol)
  fr11 = Descriptors.fr_NH0(mol)
  fr12 = Descriptors.fr_NH1(mol)
  fr13 = Descriptors.fr_NH2(mol)
  fr14 = Descriptors.fr_N_O(mol)
  fr15 = Descriptors.fr_Nhpyrrole(mol)
  fr16 = Descriptors.fr_SH(mol)
  fr17 = Descriptors.fr_alkyl_halide(mol)
  fr18 = Descriptors.fr_amide(mol)
  fr19 = Descriptors.fr_amidine(mol)
  fr20 = Descriptors.fr_azide(mol)
  fr21 = Descriptors.fr_azo(mol)
  fr22 = Descriptors.fr_bicyclic(mol)
  fr23 = Descriptors.fr_diazo(mol)
  fr24 = Descriptors.fr_dihydropyridine(mol)
  fr25 = Descriptors.fr_ester(mol)
  fr26 = Descriptors.fr_furan(mol)
  fr27 = Descriptors.fr_hdrzine(mol)
  fr28 = Descriptors.fr_hdrzone(mol)
  fr29 = Descriptors.fr_imide(mol)
  fr30 = Descriptors.fr_morpholine(mol)
  fr31 = Descriptors.fr_nitrile(mol)
  fr32 = Descriptors.fr_nitro(mol)
  fr33 = Descriptors.fr_nitro_arom(mol)
  fr34 = Descriptors.fr_nitro_arom_nonortho(mol)
  fr35 = Descriptors.fr_nitroso(mol)
  fr36 = Descriptors.fr_oxazole(mol)
  fr37 = Descriptors.fr_phenol(mol)
  fr38 = Descriptors.fr_phenol_noOrthoHbond(mol)
  fr39 = Descriptors.fr_phos_acid(mol)
  fr40 = Descriptors.fr_phos_ester(mol)
  fr41 = Descriptors.fr_piperdine(mol)
  fr42 = Descriptors.fr_piperzine(mol)
  fr43 = Descriptors.fr_priamide(mol)
  fr44 = Descriptors.fr_prisulfonamd(mol)
  fr45 = Descriptors.fr_pyridine(mol)
  fr46 = Descriptors.fr_quatN(mol)
  fr47 = Descriptors.fr_sulfide(mol)
  fr48 = Descriptors.fr_sulfonamd(mol)
  fr49 = Descriptors.fr_sulfone(mol)
  fr50 = Descriptors.fr_thiazole(mol)
  fr51 = Descriptors.fr_thiophene(mol)
  fr52 = Descriptors.fr_unbrch_alkane(mol)
  fr53 =	fr_quinoline(mol)
  fr54 =	fr_cyclpropnitro(mol)
  fr55 =	fr_Thiadiazole(mol)
  fr56 =	fr_dioxolane(mol)
  fr57 =	fr_Pyrazinedihydro(mol)
  fr58 =	fr_pyridine(mol)
  fr59 =	fr_betalactone(mol)
  fr60 =	fr_nitrosomorpholine(mol)
  fr61 =	fr_dichlorodioxane(mol)
  fr62 =	fr_dioxane(mol)
  fr63 =	fr_metTHF(mol)
  fr64 =	fr_4pyranone(mol)
  fr65 =	fr_THF(mol)
  fr66 =	fr_Dimethoxyethane(mol)
  fr67 =	fr_Dihydropyran(mol)
  fr68 =	fr_Tetrahydropyran(mol)
  fr69 =	fr_Isoxazole(mol)
  fr70 =	fr_Thiazole(mol)
  fr71 =	fr_Pyridazine(mol)
  fr72 =	fr_Pyrimidine(mol)
  fr73 =	fr_Pyrazine(mol)
  fr74 =	fr_Dimethylisoxazole(mol)
  fr75 =	fr_Oxetane(mol)
  fr76 =	fr_Arginine(mol)
  fr77 =	fr_Proline(mol)
  fr78 =	fr_Tryptophane(mol)
  fr79 =	fr_Alanine(mol)
  fr80 =	fr_Lysine(mol)
  fr81 =	fr_Phenylalanine(mol)
  fr82 =	fr_Tyrosine(mol)
  fr83 =	fr_Methionine(mol)
  fr84 =	fr_Leucine(mol)
  fr85 =	fr_Isoleucine(mol)
  fr86 =	fr_Valine(mol)
  fr87 =	fr_Glutamate(mol)
  fr88 =	fr_Glutamine(mol)
  fr89 =	fr_Aspartate(mol)
  fr90 =	fr_Glycine(mol)
  fr91 =	fr_Histidine(mol)
  fr92 =	fr_Serine(mol)
  fr93 =	fr_Threonine(mol)
  fr94 =	fr_Asparagine(mol)
  fr95 =	fr_Cysteine(mol)
  fr96 =	fr_Chromene(mol)
  fr97 =	fr_Chromene_2(mol)
  fr98 =	fr_Chromane(mol)
  fr99 = fr_Chromanone(mol)
  fr100 =fr_Chromone_2(mol)
  fr101 =fr_Furan_2(mol)
  fr102 =fr_Oxazoline(mol)
  fr103 =fr_Nitrobenzene(mol)
  fr104 =fr_Thiophene_N(mol)
  fr105 =fr_Quinolonium(mol)
  fr106 =fr_Benzimidazole(mol)
  fr107 =fr_Chlorzoxazone(mol)
  fr108 =fr_Naphthalene(mol)
  fr109 = fr_so2(mol)
  fr110 = fr_Trithiolane(mol)                          
  fr111 = fr_Dithiolane(mol)
  fr112 = fr_DiSulf(mol)
  fr113 = fr_Chlorobenzene(mol)
  fr114 = fr_dichlorobenzene12(mol)
  fr115 = fr_dichlorobenzene14(mol)
  fr118 = fr_propanoic_SMART(mol)
  fr119 = fr_propanoic(mol)
  fr120 = fr_urea(mol)
  #count
  ct149 = ct_phosphorous(mol)
  ct121 = ct_quatNwC(mol)
  ct122 = ct_Ammoniopropyl(mol)
  ct123 = ct_Aniline(mol)
  ct124 = ct_ACA(mol)
  ct125 = ct_AN(mol)
  ct126 = ct_ANH(mol)
  ct127 = ct_COdbO(mol)
  ct128 = ct_CdbNdbO(mol)
  ct129 = ct_CdbN(mol)
  ct130 = ct_CN(mol)
  ct131 = ct_Etsub(mol)
  ct132 = ct_ringNO(mol)
  ct133 = ct_Etplus(mol)
  ct134 = ct_benzoicsulfonic(mol)
  ct135 = ct_Etroot(mol)
  ct136 = ct_Benzoxazolium(mol)
  ct137 = ct_Benzoxazol(mol)
  ct138 = ct_Dapi(mol)
  ct139 = ct_HemiBabim(mol)
  ct140 = ct_HemiBabim_2(mol)
  ct53 =	ct_quinoline(mol)
  ct54 =	ct_cyclpropnitro(mol)
  ct55 =	ct_Thiadiazole(mol)
  ct56 =	ct_dioxolane(mol)
  ct57 =	ct_Pyrazinedihydro(mol)
  ct58 =	ct_pyridine(mol)
  ct59 =	ct_betalactone(mol)
  ct60 =	ct_nitrosomorpholine(mol)
  ct61 =	ct_dichlorodioxane(mol)
  ct62 =	ct_dioxane(mol)
  ct63 =	ct_metTHF(mol)
  ct64 =	ct_4pyranone(mol)
  ct65 =	ct_THF(mol)
  ct66 =	ct_Dimethoxyethane(mol)
  ct67 =	ct_Dihydropyran(mol)
  ct68 =	ct_Tetrahydropyran(mol)
  ct69 =	ct_Isoxazole(mol)
  ct70 =	ct_Thiazole(mol)
  ct71 =	ct_Pyridazine(mol)
  ct72 =	ct_Pyrimidine(mol)
  ct73 =	ct_Pyrazine(mol)
  ct74 =	ct_Dimethylisoxazole(mol)
  ct75 =	ct_Oxetane(mol)
  ct76 =	ct_Arginine(mol)
  ct77 =	ct_Proline(mol)
  ct78 =	ct_Tryptophane(mol)
  ct79 =	ct_Alanine(mol)
  ct80 =	ct_Lysine(mol)
  ct81 =	ct_Phenylalanine(mol)
  ct82 =	ct_Tyrosine(mol)
  ct83 =	ct_Methionine(mol)
  ct84 =	ct_Leucine(mol)
  ct85 =	ct_Isoleucine(mol)
  ct86 =	ct_Valine(mol)
  ct87 =	ct_Glutamate(mol)
  ct88 =	ct_Glutamine(mol)
  ct89 =	ct_Aspartate(mol)
  ct90 =	ct_Glycine(mol)
  ct91 =	ct_Histidine(mol)
  ct92 =	ct_Serine(mol)
  ct93 =	ct_Threonine(mol)
  ct94 =	ct_Asparagine(mol)
  ct95 =	ct_Cysteine(mol)
  ct96 =	ct_Chromene(mol)
  ct97 =	ct_Chromene_2(mol)
  ct98 =	ct_Chromane(mol)
  ct99 =  ct_Chromanone(mol)
  ct100 = ct_Chromone_2(mol)
  ct101 = ct_Furan_2(mol)
  ct102 = ct_Oxazoline(mol)
  ct103 = ct_Nitrobenzene(mol)
  ct104 = ct_Thiophene_N(mol)
  ct105 = ct_Quinolonium(mol)
  ct106 = ct_Benzimidazole(mol)
  ct107 = ct_Chlorzoxazone(mol)
  ct108 = ct_Naphthalene(mol)
  ct109 = ct_so2(mol)
  ct110 = ct_Trithiolane(mol)                          
  ct111 = ct_Dithiolane(mol)
  ct112 = ct_DiSulf(mol)
  ct113 = ct_Chlorobenzene(mol)
  ct114 = ct_dichlorobenzene12(mol)
  ct115 = ct_dichlorobenzene14(mol)
  ct119 = ct_propanoic(mol)
  ct120 = ct_urea(mol)
  ct_soneutral = ct_so2neutral(mol)
  ct_phosnegative = ct_phosphorous_negative(mol)
  ct_fluorbenzene = ct_fluorobenzene(mol)

  X = prints 
  Y = ([nNO, nAromaticRing, Carbocycles, HeteroRing, ct_fluorbenzene, ct_phosnegative, ct_soneutral, fr141, fr142, fr143, fr144, fr145, fr146, fr147, fr148, ct149, ct121, ct122, ct123, ct124, ct125, ct126, ct127, ct128, ct129, ct130, ct131, ct132, ct133, ct134, ct135, ct136, ct137, ct138, ct139, ct140, ct53, ct54, ct55, ct56, ct57, ct58, ct59, ct60, ct61, ct62, ct63, ct64, ct65, ct66, ct67, ct68, ct69, ct70, ct71, ct72, ct73, ct74, ct75, ct76, ct77, ct78, ct79, ct80, ct81, ct82, ct83, ct84, ct85, ct86, ct87, ct88, ct89, ct90, ct91, ct92, ct93, ct94, ct95, ct96, ct97, ct98, ct99, ct100, ct101, ct102, ct103, ct104, ct105, ct106, ct107, ct108, ct109, ct110, ct111, ct112, ct113, ct114, ct115, ct119, ct120, fr121, fr122, fr123, fr124, fr125, fr126, fr127, fr128, fr129, fr130, fr131, fr132, fr133, fr134, fr135, fr136, fr137, fr138, fr139, fr140, fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fr12, fr13, fr14, fr15, fr16, fr17, fr18, fr19, fr20, fr21, fr22, fr23, fr24, fr25, fr26, fr27, fr28, fr29, fr30, fr31, fr32, fr33, fr34, fr35, fr36, fr37, fr38, fr39, fr40, fr41, fr42, fr43, fr44, fr45, fr46, fr47, fr48, fr49, fr50, fr51, fr52, fr53, fr54, fr55, fr56, fr57, fr58, fr59, fr60, fr61, fr62, fr63, fr64, fr65, fr66, fr67, fr68, fr69, fr70, fr71, fr72, fr73, fr74, fr75, fr76, fr77, fr78, fr79, fr80, fr81, fr82, fr83, fr84, fr85, fr86, fr87, fr88, fr89, fr90, fr91, fr92, fr93, fr94, fr95, fr96, fr97, fr98, fr99, fr100, fr101, fr102, fr103, fr104, fr105, fr106, fr107, fr108, fr109, fr110, fr111, fr112, fr113, fr114, fr115, fr118, fr119, fr120])
  return np.append(X,Y)

df = pd.DataFrame()

df['Descriptors'] = fps_plus_others(mol)

X_new = np.array(list(df['Descriptors']))

X_new = X_new.reshape(1, -1)

y_pred = model.predict(X_new, verbose=1)

print("KarLogP = ", y_pred)

