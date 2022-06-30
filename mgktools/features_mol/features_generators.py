#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable, List, Union
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


class FeaturesGenerator:
    def __init__(self, features_generator_name: str,
                 radius: int = 2,
                 num_bits: int = 2048):
        self.features_generator_name = features_generator_name
        self.radius = radius
        self.num_bits = num_bits

    def __call__(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        if self.features_generator_name == 'morgan':
            return self.morgan_binary_features_generator(mol)
        elif self.features_generator_name == 'morgan_count':
            return self.morgan_counts_features_generator(mol)
        elif self.features_generator_name == 'rdkit_2d':
            return self.rdkit_2d_features_generator(mol)
        elif self.features_generator_name == 'rdkit_2d_normalized':
            return self.rdkit_2d_normalized_features_generator(mol)
        else:
            raise ValueError(f'unknown features generator: {self.features_generator_name}')

    def get_features_generator(self) -> Callable[[Union[str, Chem.Mol]], np.ndarray]:
        if self.features_generator_name == 'morgan':
            return self.morgan_binary_features_generator

    def morgan_binary_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates a binary Morgan fingerprint for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :param radius: Morgan fingerprint radius.
        :param num_bits: Number of bits in Morgan fingerprint.
        :return: A 1D numpy array containing the binary Morgan fingerprint.
        """
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.num_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)

        return features

    def morgan_counts_features_generator(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates a counts-based Morgan fingerprint for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :param radius: Morgan fingerprint radius.
        :param num_bits: Number of bits in Morgan fingerprint.
        :return: A 1D numpy array containing the counts-based Morgan fingerprint.
        """
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        features_vec = AllChem.GetHashedMorganFingerprint(mol, self.radius, nBits=self.num_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)

        return features

    @staticmethod
    def rdkit_2d_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates RDKit 2D features_mol for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features_mol.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]

        return features

    @staticmethod
    def rdkit_2d_normalized_features_generator(mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Generates RDKit 2D normalized features_mol for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features_mol.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features
