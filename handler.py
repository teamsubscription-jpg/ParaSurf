"""
RunPod serverless handler for ParaSurf (antibody paratope prediction).

Example job input:
    {"input": {"pdb": "<contents of an antibody .pdb file>"}}
    {"input": {"pdb": "...", "model": "pecan", "mesh_dense": 0.1, "return_pdb": false}}

`model` is one of: paragraph_expanded (default), pecan, heavy, light.
"""
import os
import shutil
import tempfile
import uuid
import warnings

import numpy as np
import runpod
import torch

from ParaSurf.preprocess.clean_dataset import clean_dataset
from ParaSurf.train.protein import Protein_pred
from ParaSurf.train.network import Network
from ParaSurf.train.bsite_extraction import Bsite_extractor
from ParaSurf.train.utils import receptor_info, write_residue_prediction_pdb, write_atom_prediction_pdb, antibody_input_recognition

warnings.filterwarnings('ignore')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.environ.get('PARASURF_WEIGHTS_DIR', os.path.join(APP_DIR, 'weights'))
MODELS = {
    'paragraph_expanded': 'Paragraph_expanded_best.pth',
    'pecan': 'Pecan_best.pth',
    'heavy': 'Paragraph_expanded_heavy_best.pth',
    'light': 'Paragraph_expanded_light_best.pth',
}

CFG_blind_pred = {
    'batch_size': 64,
    'Grid_size': 41,  # size of the voxel
    'feature_channels': 22,
    'add_atoms_radius_ff_features': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

_networks = {}


def get_network(model):
    if model not in _networks:
        _networks[model] = Network(os.path.join(WEIGHTS_DIR, MODELS[model]), gridSize=CFG_blind_pred['Grid_size'],
                                   feature_channels=CFG_blind_pred['feature_channels'], device=CFG_blind_pred['device'])
    return _networks[model]


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def predict(receptor, model, mesh_dense, return_pdb):
    """Same pipeline as blind_predict.py, returning the results instead of only writing files."""
    work_dir = os.path.dirname(receptor)
    rec_name = os.path.basename(receptor).split('.')[0]
    results_save_path = os.path.join(work_dir, rec_name)

    is_antibody, message = antibody_input_recognition(receptor)
    if not is_antibody:
        return {'error': f'Validation Failed: {message} Please provide a valid antibody structure as input.'}

    clean_dataset(work_dir)

    prot = Protein_pred(receptor, save_path=work_dir, mesh_dense=mesh_dense)
    surf_files = [i for i in os.listdir(prot.save_path) if 'surfpoints' in i]
    if not surf_files:
        return {'error': 'DMS failed to generate the surface points for this structure.'}
    surf_file = os.path.join(prot.save_path, surf_files[0])

    only_receptor_atoms_indexes = []
    with open(surf_file, 'r') as file:
        for atom_id, line in enumerate(file):
            parts = line.split()
            if parts[6] == 'A':
                only_receptor_atoms_indexes.append(atom_id)

    nn = get_network(model)
    lig_scores = nn.get_lig_scores(prot, batch_size=CFG_blind_pred['batch_size'], add_forcefields=True,
                                   add_atom_radius_features=CFG_blind_pred['add_atoms_radius_ff_features'])
    lig_scores_only_receptor_atoms = np.array([lig_scores[i] for i in only_receptor_atoms_indexes])

    residues, residues_best = receptor_info(receptor, lig_scores_only_receptor_atoms)

    write_residue_prediction_pdb(receptor, results_save_path, residues_best)
    write_atom_prediction_pdb(receptor, results_save_path, lig_scores_only_receptor_atoms)

    extractor = Bsite_extractor()
    extractor.extract_bsites(prot, lig_scores)

    residue_scores = []
    for res_id, res_data in residues_best.items():
        parts = res_id.split('_')
        residue_scores.append({
            'residue_id': res_id,
            'chain': parts[1],
            'residue_number': int(parts[0]),
            'insertion_code': parts[2] if len(parts) > 2 else '',
            'score': float(res_data['scores']),
        })
    residue_scores.sort(key=lambda r: r['score'], reverse=True)

    output = {'model': model, 'device': CFG_blind_pred['device'], 'residues': residue_scores}
    if return_pdb:
        output['pred_pdb'] = read_file(os.path.join(results_save_path, f'{rec_name}_pred.pdb'))
        output['pred_per_atom_pdb'] = read_file(os.path.join(results_save_path, f'{rec_name}_pred_per_atom.pdb'))
        pockets = sorted(f for f in os.listdir(prot.save_path) if f.startswith('pocket') and f.endswith('.pdb'))
        output['pockets'] = [read_file(os.path.join(prot.save_path, f)) for f in pockets]
    return output


def handler(job):
    job_input = job.get('input', {})
    pdb = job_input.get('pdb')
    model = job_input.get('model', 'paragraph_expanded')
    return_pdb = bool(job_input.get('return_pdb', True))

    if not pdb or not isinstance(pdb, str):
        return {'error': "Provide 'pdb': the contents of an antibody PDB file."}
    if model not in MODELS:
        return {'error': f"Invalid model: {model}. Choose from {', '.join(MODELS)}."}
    try:
        mesh_dense = float(job_input.get('mesh_dense', 0.3))
    except (TypeError, ValueError):
        return {'error': 'mesh_dense must be a number between 0.1 and 1.0'}
    if not (0.1 <= mesh_dense <= 1.0):
        return {'error': 'mesh_dense must be between 0.1 and 1.0'}

    # ParaSurf derives file names from the path (splitting on '.' and '_'), so keep both out of it
    work_dir = tempfile.mkdtemp(prefix='parasurf', dir='/tmp')
    receptor = os.path.join(work_dir, f'rec{uuid.uuid4().hex}.pdb')
    with open(receptor, 'w') as f:
        f.write(pdb)

    try:
        return predict(receptor, model, mesh_dense, return_pdb)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
