import csv
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from ..datasets.trajectory_dataset import TrajectoryDataset
from ..engine.losses import get_loss
from ..engine.metrics import mean_l2_error, median_l2_error, success_rate_at_threshold
from ..utils.seed import set_seed


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_model(cfg):
    mcfg = cfg['model']
    name = mcfg.get('name', 'single_frame')
    if name == 'single_frame':
        from ..models.single_frame_regressor import SingleFrameRegressor
        model = SingleFrameRegressor(encoder_channels=mcfg.get('encoder_channels', [16,32,64]), hidden_dim=mcfg.get('hidden_dim',128))
    else:
        from ..models.cnn_gru_regressor import CNNGRURegressor
        model = CNNGRURegressor(encoder_channels=mcfg.get('encoder_channels',[16,32,64]), rnn_hidden=mcfg.get('rnn_hidden',128), rnn_layers=mcfg.get('rnn_layers',1))
    return model


def build_dataset(ds_cfg, split_key):
    split_file = ds_cfg.get(f'{split_key}_split_file') or ds_cfg.get('split_file')
    return TrajectoryDataset(
        root=ds_cfg['root'],
        split_file=split_file,
        img_size=ds_cfg.get('img_size', 64),
        observation_length=ds_cfg.get('observation_length', 8),
        frame_stride=ds_cfg.get('frame_stride', 1),
        use_last_n_frames=ds_cfg.get('use_last_n_frames', True),
        observation_end_fraction=ds_cfg.get('observation_end_fraction', 1.0),
        sampling_mode=ds_cfg.get('sampling_mode', 'tail'),
    )


def evaluate_loader(model, loader, device, loss_fn, success_thresholds=None):
    model.eval()
    losses = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            targets = batch['target_xy'].to(device)
            preds = model(frames)
            loss = loss_fn(preds, targets)
            losses.append(loss.item())
            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    metrics = {}
    if preds_all:
        preds_np = np.concatenate(preds_all, axis=0)
        targets_np = np.concatenate(targets_all, axis=0)
        metrics = {
            'mean_l2': mean_l2_error(preds_np, targets_np),
            'median_l2': median_l2_error(preds_np, targets_np),
        }
        for threshold in success_thresholds or []:
            metrics[f'success_at_{threshold}'] = success_rate_at_threshold(preds_np, targets_np, threshold)

    return float(np.mean(losses)) if losses else 0.0, metrics


def save_history_csv(history, path):
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'val_mean_l2', 'val_median_l2', 'val_success']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_curve_svg(history, path, title, y_label, keys_and_colors):
    if not history:
        return

    width = 800
    height = 480
    margin_left = 70
    margin_right = 30
    margin_top = 30
    margin_bottom = 55
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    epochs = [row['epoch'] for row in history]
    y_values = []
    present_series = []
    for key, color, label in keys_and_colors:
        series = [row[key] for row in history if row[key] is not None]
        if series:
            y_values.extend(series)
            present_series.append((key, color, label))
    if not y_values:
        return
    y_min = min(y_values)
    y_max = max(y_values)
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0

    def x_pos(epoch):
        if len(epochs) == 1:
            return margin_left + plot_w / 2.0
        return margin_left + (epoch - epochs[0]) / (epochs[-1] - epochs[0]) * plot_w

    def y_pos(value):
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    def polyline(values):
        pts = []
        for row in history:
            value = row[values]
            if value is None:
                continue
            pts.append(f"{x_pos(row['epoch']):.2f},{y_pos(value):.2f}")
        return " ".join(pts)

    x_ticks = sorted(set([epochs[0], epochs[-1], max(1, epochs[-1] // 2)]))
    y_ticks = np.linspace(y_min, y_max, num=5)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial, sans-serif;font-size:12px;fill:#222} .grid{stroke:#ddd;stroke-width:1} .axis{stroke:#333;stroke-width:1.5} .title{font-size:18px;font-weight:bold}</style>',
        f'<text x="{width/2:.0f}" y="20" text-anchor="middle" class="title">{title}</text>',
    ]

    for tick in y_ticks:
        y = y_pos(float(tick))
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" class="grid" />')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.2f}" text-anchor="end">{tick:.4f}</text>')

    for tick in x_ticks:
        x = x_pos(tick)
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height-margin_bottom}" class="grid" />')
        lines.append(f'<text x="{x:.2f}" y="{height-margin_bottom+20}" text-anchor="middle">{tick}</text>')

    lines.extend([
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" class="axis" />',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" class="axis" />',
        f'<text x="{width/2:.0f}" y="{height-15}" text-anchor="middle">Epoch</text>',
        f'<text x="18" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.0f})">{y_label}</text>',
    ])

    legend_x = width - margin_right - 140
    legend_y = margin_top + 10
    for idx, (key, color, label) in enumerate(present_series):
        class_name = f'series{idx}'
        lines.insert(1, f'<style>.{class_name}{{fill:none;stroke:{color};stroke-width:2.5}}</style>')
        lines.append(f'<polyline points="{polyline(key)}" class="{class_name}" />')
        lines.extend([
            f'<line x1="{legend_x}" y1="{legend_y + 24 * idx}" x2="{legend_x+24}" y2="{legend_y + 24 * idx}" class="{class_name}" />',
            f'<text x="{legend_x+32}" y="{legend_y + 24 * idx + 4}">{label}</text>',
        ])

    lines.append('</svg>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def train_from_config(cfg_path):
    cfg = load_config(cfg_path)
    ds_cfg = cfg['dataset']
    tr_cfg = cfg['training']
    out_dir = tr_cfg.get('output_dir','temperal/outputs/exp')
    os.makedirs(out_dir, exist_ok=True)
    set_seed(int(tr_cfg.get('seed', 42)))

    dataset = build_dataset(ds_cfg, 'train')
    loader = DataLoader(dataset, batch_size=int(tr_cfg.get('batch_size',8)), shuffle=True, num_workers=0)
    val_loader = None
    if ds_cfg.get('val_split_file'):
        val_dataset = build_dataset(ds_cfg, 'val')
        val_loader = DataLoader(val_dataset, batch_size=int(tr_cfg.get('batch_size',8)), shuffle=False, num_workers=0)

    model = build_model(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() and tr_cfg.get('device','auto')=='auto' else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr_cfg.get('lr',1e-3)), weight_decay=float(tr_cfg.get('weight_decay',0.0)))
    loss_fn = get_loss()
    success_thresholds = cfg.get('eval', {}).get('thresholds', [0.2])
    primary_threshold = success_thresholds[0] if success_thresholds else 0.2

    epochs = int(tr_cfg.get('epochs',5))
    best_val = float('inf')
    history = []

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for batch in loader:
            frames = batch['frames'].to(device)
            targets = batch['target_xy'].to(device)
            preds = model(frames)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        row = {
            'epoch': ep,
            'train_loss': avg_loss,
            'val_loss': None,
            'val_mean_l2': None,
            'val_median_l2': None,
            'val_success': None,
        }

        log_line = f"Epoch {ep}/{epochs} train_loss={avg_loss:.4f}"
        if val_loader is not None:
            val_loss, val_metrics = evaluate_loader(model, val_loader, device, loss_fn, success_thresholds=success_thresholds)
            row['val_loss'] = val_loss
            row['val_mean_l2'] = val_metrics.get('mean_l2')
            row['val_median_l2'] = val_metrics.get('median_l2')
            row['val_success'] = val_metrics.get(f'success_at_{primary_threshold}')
            log_line += f" val_loss={val_loss:.4f}"
            if row['val_mean_l2'] is not None:
                log_line += f" val_mean_l2={row['val_mean_l2']:.4f}"
            if row['val_success'] is not None:
                log_line += f" val_success@{primary_threshold}={row['val_success']:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model': model.state_dict(), 'cfg': cfg}, os.path.join(out_dir, 'best.pt'))

        history.append(row)
        print(log_line)

        ckpt = {'model': model.state_dict(), 'cfg': cfg}
        torch.save(ckpt, os.path.join(out_dir, 'last.pt'))

    save_history_csv(history, os.path.join(out_dir, 'history.csv'))
    save_curve_svg(
        history,
        os.path.join(out_dir, 'loss_curve.svg'),
        title='Loss Curve',
        y_label='Loss',
        keys_and_colors=[
            ('train_loss', '#2563eb', 'train_loss'),
            ('val_loss', '#dc2626', 'val_loss'),
        ],
    )
    save_curve_svg(
        history,
        os.path.join(out_dir, 'success_curve.svg'),
        title=f'Success Curve @ {primary_threshold}',
        y_label='Success Rate',
        keys_and_colors=[
            ('val_success', '#059669', f'val_success@{primary_threshold}'),
        ],
    )
    print('Training finished. Model saved to', out_dir)
