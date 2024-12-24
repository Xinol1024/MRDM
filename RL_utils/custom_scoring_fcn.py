import numpy as np
import pandas as pd
import math
import gzip
import pickle

#import xgboost as xgb

from RL_utils.scoring_function import MoleculewiseScoringFunction
from RL_utils.scoring_function import ArithmeticMeanScoringFunction
from RL_utils.scoring_function import ScoringFunctionBasedOnRdkitMol
from RL_utils.scoring_function import MinMaxGaussianModifier

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Crippen, Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors

from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Mol
from rdkit.six import iteritems
import _pickle as cPickle


class LigandEfficancy(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, score_modifier, model_path):
        super().__init__(score_modifier=score_modifier)
        with open(model_path,'rb') as handle:
            self.rfr = pickle.load(handle)
        
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        m = Chem.MolFromSmiles(smiles)
        mph = Chem.AddHs(m)
        N = mph.GetNumAtoms() - mph.GetNumHeavyAtoms()
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2)
        fp = np.array([fp])
        pic50 = self.rfr.predict(fp)
        LE = 1.4*(pic50)/N
        return LE[0]

class QED(MoleculewiseScoringFunction):
    def __init__(self, score_modifier):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        qed = Descriptors.qed(mol)
        return qed

class SAScorer(MoleculewiseScoringFunction):
    def __init__(self, score_modifier, fscores=None):
        super().__init__(score_modifier=score_modifier)
        if fscores is None:
            fscores = '../../data/fpscores.pkl.gz'
        self.fscores = cPickle.load(gzip.open(fscores ))
        outDict = {}
        for i in self.fscores:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        self.fscores = outDict
    
    def numBridgeheadsAndSpiro(self, mol, ri=None):
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro


    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        m = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprint(m,2)  # <- 2 is the *radius* of the circular fingerprint
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in iteritems(fps):
            nf += v
            sfp = bitId
            score1 += self.fscores.get(sfp, -4) * v
        score1 /= nf
        # features score
        nAtoms = m.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        ri = m.GetRingInfo()
        nBridgeheads, nSpiro = self.numBridgeheadsAndSpiro(m, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1
        
        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)
        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5
        sascore = score1 + score2 + score3
        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0
        return sascore
        #return self.threshold/np.maximum(sascore, self.threshold)

class LogP(MoleculewiseScoringFunction):
    def __init__(self, score_modifier=None):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        logp = Descriptors.MolLogP(mol)
        return logp

class MW(MoleculewiseScoringFunction):
    def __init__(self, score_modifier=None):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.ExactMolWt(mol)
        return mw
        # try:
        #     mw=  Descriptors.ExactMolWt( Chem.MolFromSmiles(smiles) )
        #     return mw
        # except:
        #     print('we cant calculate molecular weight', smiles )
        #     return -1.


class lipinski(MoleculewiseScoringFunction):
    def __init__(self, score_modifier=None):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        rule_1 = Descriptors.ExactMolWt(mol) < 500
        rule_2 = Lipinski.NumHDonors(mol) <= 5
        rule_3 = Lipinski.NumHAcceptors(mol) <= 10
        rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])