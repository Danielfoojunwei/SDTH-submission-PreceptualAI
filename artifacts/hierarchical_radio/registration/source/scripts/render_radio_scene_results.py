"""Render an actual TRAIN CAD/report cache without inventing a signal field.

The figure shows discrete simulated device reports. It performs no fitting,
interpolation, held-out evaluation or neural inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mesh_box(path):
    lines = path.read_text().splitlines()
    start = lines.index('end_header') + 1
    count = int(next(line for line in lines if line.startswith('element vertex ')).split()[-1])
    vertices = np.array([[float(x) for x in line.split()] for line in lines[start:start + count]])
    return vertices.min(0), vertices.max(0)


def faces(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    return [v[list(indices)] for indices in ((0,1,2,3),(4,5,6,7),(0,1,5,4),
                                              (2,3,7,6),(1,2,6,5),(0,3,7,4))]


def main():
    folder = ROOT / 'artifacts/radiojepa_spectral/rt_data/training/spectral-training-00'
    receipt = json.loads((folder / 'receipt.json').read_text())
    public = folder / 'public_views.npz'
    assert sha(public) == receipt['files_sha256']['public_views.npz']
    with np.load(public, allow_pickle=False) as saved:
        rx, ap = saved['rx_positions_m'], saved['ap_positions_m']
        values = saved['rssi_dbm'][:, 0, :2, 0, 0]
        heard = saved['heard'][:, 0, :2, 0, 0]
        censored = saved['censored'][:, 0, :2, 0, 0]
    destination = ROOT / 'artifacts/scene_field_figures'
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#e5ecf5',
                         'axes.labelcolor':'#cbd5e1','xtick.color':'#94a3b8',
                         'ytick.color':'#94a3b8'})
    fig = plt.figure(figsize=(14, 7), facecolor='#08111f', constrained_layout=True)
    scale = plt.Normalize(-90, -30)
    inputs = {'receipt.json': sha(folder / 'receipt.json'), 'public_views.npz': sha(public)}
    for band, title in enumerate(('2.4 GHz', '5.2 GHz')):
        ax = fig.add_subplot(1, 2, band + 1, projection='3d', facecolor='#08111f')
        for path in sorted((folder / 'scene/meshes').glob('*.ply')):
            relative = str(path.relative_to(folder))
            assert sha(path) == receipt['files_sha256'][relative]
            inputs[relative] = sha(path)
            lo, hi = mesh_box(path)
            boundary = path.stem.startswith(('floor','ceiling','wall'))
            ax.add_collection3d(Poly3DCollection(faces(lo,hi),
                facecolors='#3c5878' if boundary else '#83a4c8',
                edgecolors='#607b9a', linewidths=.35, alpha=.035 if boundary else .23))
        known = heard[:, band]
        ax.scatter(*rx[known].T, c=values[known, band], cmap='turbo', norm=scale,
                   s=95, depthshade=False, edgecolors='#d8e5f5', linewidths=.6)
        unknown = ~known & ~censored[:,band]
        if unknown.any():
            ax.scatter(*rx[unknown].T, marker='x', c='#94a3b8', s=45, label='unresolved')
        if censored[:,band].any():
            ax.scatter(*rx[censored[:,band]].T, marker='v', c='#e5ecf5', s=55, label='censored')
        ax.scatter(*ap.T, marker='^', s=160, c='#fbbf24', depthshade=False)
        for i, point in enumerate(ap):
            ax.text(*point, f'  AP {i}', color='#fbbf24', fontsize=9)
        for i, point in enumerate(rx):
            ax.text(*(point + np.array([0,0,.12])), str(i), fontsize=7, color='#d8e5f5')
        dims = np.array([receipt['spec'][x] for x in ('width_m','depth_m','ceiling_m')])
        ax.set(xlim=(-dims[0]/2,dims[0]/2), ylim=(-dims[1]/2,dims[1]/2),
               zlim=(0,dims[2]), xlabel='x (m)', ylabel='y (m)', zlabel='height (m)')
        ax.set_box_aspect(dims)
        ax.view_init(elev=27, azim=-58)
        ax.set_title(title, color='#f1f5f9', fontsize=16, pad=12)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo['grid']['color'] = '#26384c'
    mappable = plt.cm.ScalarMappable(norm=scale, cmap='turbo')
    colorbar = fig.colorbar(mappable, ax=fig.axes, shrink=.5, pad=.02)
    colorbar.set_label('Simulated reported RSSI to AP 0 (dBm)', color='#cbd5e1')
    colorbar.ax.tick_params(colors='#94a3b8')
    fig.suptitle('Registered TRAIN scene · actual Sionna CAD and discrete device reports',
                 fontsize=17, color='#f8fafc')
    fig.text(.02, .02, 'Markers are observations; no interpolated field or inferred paths.  '
             'Even poses: context.  Odd poses: training query labels.', color='#94a3b8', fontsize=9)
    output = destination / 'training-scene.png'
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    (destination / 'receipt.json').write_text(json.dumps({
        'scope':'actual simulated TRAIN CAD and reported RSSI, not hardware or an inferred field',
        'source_scene':str(folder.relative_to(ROOT)), 'source_sha256':inputs,
        'script_sha256':sha(Path(__file__)), 'output_sha256':sha(output),
        'new_fits_or_interpolated_measurements':False}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
