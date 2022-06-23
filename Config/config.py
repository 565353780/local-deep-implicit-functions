#!/usr/bin/env python
# -*- coding: utf-8 -*-

TRAIN_CONFIG = {
    'experiment_name': 'test_1',
    'model_directory': 'trained_models/',
    'dataset_directory': '',
    'model_type': 'ldif',
    "batch_size": 1,
    'summary_step_interval': 10,
    'checkpoint_interval': 250,
    'train_step_count': 1000000,
    'split': 'train',
    'visualize': False,
    #  'reserve_memory_for_inference_kernel': True,
}

