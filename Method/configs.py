#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ldif.model import hparams
from ldif.inference import experiment as experiments

from Config.config import TRAIN_CONFIG

def build_model_config(dataset):
  # TODO(kgenova) This needs to somehow at least support LDIF/SIF/SingleView.
  # TODO(kgenova) Add support for eval/inference.

  builder_fun_dict = {
      'ldif': hparams.build_ldif_hparams,
      'sif': hparams.build_sif_hparams,
      'sif++': hparams.build_improved_sif_hparams
  }

  model_type = TRAIN_CONFIG['model_type']
  batch_size = TRAIN_CONFIG['batch_size']
  split = TRAIN_CONFIG['split']

  model_config = experiments.ModelConfig(builder_fun_dict[model_type]())
  model_config.hparams.bs = batch_size
  model_config.train = True
  model_config.eval = False
  model_config.inference = False
  model_config.inputs['dataset'] = dataset
  model_config.inputs['split'] = split
  model_config.inputs['proto'] = 'ShapeNetNSSDodecaSparseLRGMediumSlimPC'
  model_config.wrap_optimizer = lambda x: x
  return model_config

