#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from ldif.util import path_util

from Config.config import TRAIN_CONFIG

def get_model_root():
    '''
    Finds the path to the trained model's root directory.
    '''
    ldif_abspath = path_util.get_path_to_ldif_root()

    model_directory = TRAIN_CONFIG['model_directory']
    model_dir_is_relative = model_directory[0] != '/'

    if model_dir_is_relative:
        model_dir_path = os.path.join(ldif_abspath, model_directory)
    else:
        model_dir_path = model_directory

    if not os.path.isdir(model_dir_path):
        os.makedirs(model_dir_path)
    return model_dir_path

