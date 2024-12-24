import os
# import logging
import numpy as np
import torch
from rdkit import Chem
import csv


from RL_utils.score_fuc import build_scoring_function
from RL_utils.sample_drug3d_forRL import sample
from RL_utils.train_drug3d_forRL import train_forRL
from RL_utils.docking import dock_socre


# Save molecules locally in sdf and SMILES formats
def save_sdf(sdf_list, sdf_dir):
    sdf_dir = sdf_dir
    os.makedirs(sdf_dir, exist_ok=True)
    csv_file_path = os.path.join(sdf_dir, 'mol_summary.csv')
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if os.path.getsize(csv_file_path) == 0:  # Check if the file is empty
            csv_writer.writerow(['', 'mol_id', 'smiles'])  # Add header row if file is empty
        for i, data_finished in enumerate(sdf_list):
            smiles = data_finished['smiles']
            csv_writer.writerow([i, i, smiles])  # Skip first column
            rdmol = data_finished['rdmol']
            Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, '%d.sdf' % i))

if __name__ == '__main__':
    device = torch.device("cuda:0")
    if device.type.startswith("cuda"):
        torch.cuda.set_device(device.index or 0)

    # logger = logging.getLogger()

    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Path
    scoring_definition_path = "./RL_utils/scoring_definition_MEK1_mTOR_new.csv"
    initial_model_ckpt_path = "./logs/train_MolDiff_20240506_124844/checkpoints/1600000.pt"
    initial_mol_path = "./RL_utils/out/init"

    # Define scoring function
    # Get reinforcement objective
    print("RL|Define scoring function.")

    scorers = None
    scoring_function = build_scoring_function( 
                            scoring_definition = scoring_definition_path,
                            fscores = "./RL_utils/fpscores.pkl.gz",
                            return_individual = False)

    # Define initial population
    if not os.path.exists(initial_mol_path):
        print("RL|The initial population does not exist, start generating the initial population.")
        gen_list = sample(model_ckpt = initial_model_ckpt_path, num_mols = 3000, epoch_now=0)

        # Score the molecules in the initial population
        scores_all = 0
        dock_scores_all = 0
        dock = True
        print("RL|Start docking...")
        for i in range(len(gen_list)):
            if True:
                scores = scoring_function.score_list([gen_list[i]['smiles']])
                # gen_list[i]['score'] = scores
                print("The individual score is:", scores, scores[0])
                scores_all = scores_all + scores[0]
                print(gen_list[i])
            if dock:
                normalized_score, dock_scores_all = dock_socre(gen_list, dock_scores_all, i)
                gen_list[i]['score'] = (scores + normalized_score) / 2
            else:
                gen_list[i]['score'] = scores

        if dock:
            scores_all = (dock_scores_all + scores_all) / 2
        print("RL|The average score is:", scores_all/len(gen_list))
        gen_list = sorted(gen_list, key=lambda x: x['score'], reverse=True)

        save_sdf(gen_list, initial_mol_path)
    else:
        # gen_list = []
        # print("RL|Use the existing initial population.")
        raise RuntimeError("RL|Use the existing initial population.")


    n_epochs = 50
    for epoch in range(1, n_epochs):
        print(f'RL|Starting Epoch: {epoch}')

        # Read the molecule of the previous epoch and process the dataset
        # Train
        if epoch == 1:
            ckpt_path = train_forRL(initial_mol_path, epoch, initial_model_ckpt_path)
        else:
            ckpt_path = train_forRL(save_path, epoch, ckpt_path)

        # Generate a new round of molecules using a trained model
        gen_list_new = sample(model_ckpt = ckpt_path, num_mols = 2000, epoch_now = epoch)

        # Add a new round of molecules to the generated list of molecules, estimate scores and sort them
        scores_all = 0
        dock_scores_all = 0
        dock = True
        gen_list = gen_list + gen_list_new

        print("RL|Start docking...")
        for i in range(len(gen_list)):
            if True:
                scores = scoring_function.score_list([gen_list[i]['smiles']])
                # gen_list[i]['score'] = scores
                scores_all = scores_all + scores[0]
            if dock:
                normalized_score, dock_scores_all = dock_socre(gen_list, dock_scores_all, i)
                gen_list[i]['score'] = (scores + normalized_score) / 2
            else:
                gen_list[i]['score'] = scores

        if dock:
            scores_all = (dock_scores_all + scores_all) / 2
        print("RL|The average score is:", scores_all / len(gen_list))
        gen_list = sorted(gen_list, key=lambda x: x['score'], reverse=True)
        gen_list = gen_list[:4000]
        # top_4000_scores_all = sum(item['score'][0] for item in gen_list)   
        top_4000_scores_all = sum(item['score'] for item in gen_list)   
        print("RL|The average score of the top 4000 is:", top_4000_scores_all / len(gen_list))

        # Save
        save_path = "./RL_utils/out/{}".format(epoch)
        save_sdf(gen_list, save_path)

print("Done!")