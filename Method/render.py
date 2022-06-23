#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ldif.util import gaps_util
from ldif.inference import example
from ldif.util.file_util import log

from Config.config import TRAIN_CONFIG

def visualize_data(session, dataset):
    """Visualizes the dataset with two interactive visualizer windows."""
    (bounding_box_samples, depth_renders, mesh_name, near_surface_samples, grid,
    world2grid, surface_point_samples) = session.run([
        dataset.bounding_box_samples, dataset.depth_renders, dataset.mesh_name,
        dataset.near_surface_samples, dataset.grid, dataset.world2grid,
        dataset.surface_point_samples])
    gaps_util.ptsview(
        [bounding_box_samples, near_surface_samples, surface_point_samples])
    mesh_name = mesh_name.decode(sys.getdefaultencoding())
    log.info(f'depth max: {np.max(depth_renders)}')
    log.info(f'Mesh name: {mesh_name}')
    assert '|' in mesh_name
    mesh_hash = mesh_name[mesh_name.find('|') + 1:]
    log.info(f'Mesh hash: {mesh_hash}')

    split = TRAIN_CONFIG['split']
    dyn_obj = example.InferenceExample(split, 'airplane', mesh_hash)

    gaps_util.gapsview(
        msh=dyn_obj.normalized_gt_mesh,
        pts=near_surface_samples[:, :3],
        grd=grid,
        world2grid=world2grid,
        grid_threshold=-0.07)
    return True

