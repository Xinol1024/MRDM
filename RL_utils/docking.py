import os
import subprocess
from openbabel import openbabel
from rdkit import Chem
import re
import numpy as np

class GaussianModifier:

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.))


class MinMaxGaussianModifier:

    def __init__(self, mu: float, sigma: float, minimize=False) -> None:
        self.mu = mu
        self.sigma = sigma
        self.minimize = minimize
        self._full_gaussian = GaussianModifier(mu=mu, sigma=sigma)

    def __call__(self, x):
        if self.minimize:
            mod_x = np.maximum(x, self.mu)
        else:
            mod_x = np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)


def sdf_to_pdbqt(input_sdf, output_pdbqt):

    
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdbqt")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_sdf)

    mol.AddHydrogens()  
    openbabel.OBChargeModel.FindType("gasteiger").ComputeCharges(mol)

    obConversion.WriteFile(mol, output_pdbqt)


def run_docking_and_normalize(ligand_sdf_path, receptor_pdbqt_path, center, box_size, mu, sigma, minimize=False):

    ligand_pdbqt_path = ligand_sdf_path.replace('.sdf', '.pdbqt')
    sdf_to_pdbqt(ligand_sdf_path, ligand_pdbqt_path)

    vina_command = [
        'vina', 
        '--receptor', receptor_pdbqt_path,  
        '--ligand', ligand_pdbqt_path, 
        '--center_x', str(center[0]), '--center_y', str(center[1]), '--center_z', str(center[2]),  
        '--size_x', str(box_size[0]), '--size_y', str(box_size[1]), '--size_z', str(box_size[2]),  
        '--exhaustiveness', '8', 
        '--num_modes', '1' 
    ]

    result = subprocess.run(
        vina_command,
        stdout=subprocess.PIPE,  
        stderr=subprocess.PIPE   
    )

    output = result.stdout.decode('utf-8')

    match = re.search(r'\d+\s+(-\d+\.\d+)\s+', output)
    if match:
        best_score = float(match.group(1))

        modifier = MinMaxGaussianModifier(mu=mu, sigma=sigma, minimize=minimize)
        normalized_score = modifier(best_score)
        # normalized_score = modifier(-5)
        # return normalized_score
        return normalized_score, output  
    else:
        raise ValueError("No score.")

def dock_socre(gen_list, dock_scores_all,i):
    rdmol = gen_list[i]['rdmol']
    sdf_dir = './docking_workdir/init/'
    sdf_path = os.path.join(sdf_dir, f'{i}.sdf')
    os.makedirs(sdf_dir, exist_ok=True)
    Chem.MolToMolFile(rdmol, sdf_path)
    try:
        normalized_score_mTOR, vina_output_mTOR = run_docking_and_normalize(
            ligand_sdf_path=sdf_path,
            receptor_pdbqt_path="/home/yuanyn/pxh/Diff/MRDM/RL_utils/mTOR.pdbqt",
            center=(-9.2, 26.8, 35.8),
            box_size=(126, 126, 126),
            mu=-9.0,
            sigma=1.0,
            minimize=True
        )

        normalized_score_MEK1, vina_output_MEK1 = run_docking_and_normalize(
            ligand_sdf_path=sdf_path,
            receptor_pdbqt_path="/home/yuanyn/pxh/Diff/MRDM/RL_utils/MEK1.pdbqt",
            center=(60.7, -27.0, 12.8),
            box_size=(126, 126, 126),
            mu=-9.0,
            sigma=1.0,
            minimize=True
        )

        output_txt_path = os.path.join(sdf_dir, f'mTOR_{i}_docking_results.txt')
        with open(output_txt_path, 'w') as f:
            f.write(vina_output_mTOR + "\n")
            f.write(f"Normalized docking score: {normalized_score_mTOR}\n")

        output_txt_path = os.path.join(sdf_dir, f'MEK1_{i}_docking_results.txt')
        with open(output_txt_path, 'w') as f:
            f.write(vina_output_MEK1 + "\n")
            f.write(f"Normalized docking score: {normalized_score_MEK1}\n")

    except Exception as e:
        print(f"Error occurred while processing molecule {i}: {e}")

    normalized_score = (normalized_score_mTOR + normalized_score_MEK1) / 2
    # gen_list[i]['score'] = normalized_score
    dock_scores_all = dock_scores_all + normalized_score
    return normalized_score, dock_scores_all
