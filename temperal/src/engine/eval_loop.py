import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from ..datasets.trajectory_dataset import TrajectoryDataset
from ..engine.metrics import mean_l2_error, median_l2_error, success_rate_at_threshold


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    return ck


def evaluate(cfg_path, ckpt_path, split='val'):
    with open(cfg_path,'r') as f:
        cfg = yaml.safe_load(f)
    ds_cfg = cfg['dataset']
    split_file = ds_cfg.get(f'{split}_split_file') or ds_cfg.get('eval_split_file') or ds_cfg.get('split_file')
    ds = TrajectoryDataset(root=ds_cfg['root'],
                           split_file=split_file,
                           img_size=ds_cfg.get('img_size',64),
                           observation_length=ds_cfg.get('observation_length',8),
                           frame_stride=ds_cfg.get('frame_stride',1),
                           use_last_n_frames=ds_cfg.get('use_last_n_frames',True),
                           observation_end_fraction=ds_cfg.get('observation_end_fraction', 1.0),
                           sampling_mode=ds_cfg.get('sampling_mode', 'tail'))
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() and cfg['training'].get('device','auto')=='auto' else 'cpu')
    ck = load_checkpoint(ckpt_path, device)
    model_cfg = ck.get('cfg', cfg)['model']
    name = model_cfg.get('name','single_frame')
    if name == 'single_frame':
        from ..models.single_frame_regressor import SingleFrameRegressor as M
        model = M(encoder_channels=model_cfg.get('encoder_channels',[16,32,64]), hidden_dim=model_cfg.get('hidden_dim',128))
    else:
        from ..models.cnn_gru_regressor import CNNGRURegressor as M
        model = M(encoder_channels=model_cfg.get('encoder_channels',[16,32,64]), rnn_hidden=model_cfg.get('rnn_hidden',128), rnn_layers=model_cfg.get('rnn_layers',1))
    model.load_state_dict(ck['model'])
    model.to(device)
    model.eval()

    preds_all = []
    targets_all = []
    ids = []
    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            targets = batch['target_xy'].numpy()
            out = model(frames).cpu().numpy()
            preds_all.append(out)
            targets_all.append(targets)
            ids.extend(batch['episode_id'])

    preds = np.concatenate(preds_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)

    metrics = {
        'mean_l2': mean_l2_error(preds, targets),
        'median_l2': median_l2_error(preds, targets)
    }
    thresholds = cfg.get('eval', {}).get('thresholds', [0.2])
    for t in thresholds:
        metrics[f'success_at_{t}'] = success_rate_at_threshold(preds, targets, t)

    print('Evaluation metrics:', metrics)
    return metrics
