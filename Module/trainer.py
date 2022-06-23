#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ldif.datasets import local_inputs
from ldif.model import hparams
from ldif.training import shared_launcher
from ldif.util import file_util
from ldif.util import gpu_util

from Config.config import TRAIN_CONFIG

from Method.configs import build_model_config
from Method.render import visualize_data
from Method.paths import get_model_root

class Trainer(object):
    def __init__(self):
        self.experiment_name = TRAIN_CONFIG["experiment_name"]
        self.model_directory = TRAIN_CONFIG["model_directory"]
        self.dataset_directory = TRAIN_CONFIG["dataset_directory"]
        self.model_type = TRAIN_CONFIG["model_type"]
        self.batch_size = TRAIN_CONFIG["batch_size"]
        self.summary_step_interval = TRAIN_CONFIG["summary_step_interval"]
        self.checkpoint_interval = TRAIN_CONFIG["checkpoint_interval"]
        self.train_step_count = TRAIN_CONFIG["train_step_count"]
        self.split = TRAIN_CONFIG["split"]
        self.visualize = TRAIN_CONFIG["visualize"]

        self.dataset = None
        self.model_config = None

        self.initEnv()
        return

    def initTF(self):
        tf.disable_v2_behavior()
        return True

    def loadDataset(self):
        if not os.path.exists(self.dataset_directory):
            print("[ERROR][Trainer::loadDataset]")
            print("\t dataset not exist!")
            return False

        # TODO(kgenova) This batch size should match.
        self.dataset = local_inputs.make_dataset(
            self.dataset_directory,
            mode='train',
            batch_size=self.batch_size,
            split=self.split)
        return True

    def loadModel(self):
        # Sets up the hyperparameters and tf.Dataset
        self.model_config = build_model_config(self.dataset)
        return True

    def loadSaver(self):
        self.saver = tf.train.Saver(
            max_to_keep=5, pad_step_number=False, save_relative_paths=True)
        return True

    def initEnv(self):
        if not self.initTF():
            print("[ERROR][Trainer::initEnv]")
            print("\t initTF failed!")
            return False
        if not self.loadDataset():
            print("[ERROR][Trainer::initEnv]")
            print("\t loadDataset failed!")
            return False
        if not self.loadModel():
            print("[ERROR][Trainer::initEnv]")
            print("\t loadModel failed!")
            return False
        return True

    def getGPUAllowableFraction(self):
        current_free = gpu_util.get_free_gpu_memory(0)
        allowable = current_free - (1024 + 512)  # ~1GB
        allowable_fraction = allowable / current_free
        return allowable_fraction

    def train(self):
        # Generates the graph for a single train step, including summaries
        shared_launcher.sif_transcoder(self.model_config)
        summary_op = tf.summary.merge_all()
        global_step_op = tf.compat.v1.train.get_global_step()


        init_op = tf.initialize_all_variables()

        model_root = get_model_root()

        experiment_dir = model_root + "/sif-transcoder-" + self.experiment_name
        checkpoint_dir = f'{experiment_dir}/1-hparams/train/'

        allowable_fraction = self.getGPUAllowableFraction()
        if allowable_fraction <= 0.0:
            print("[ERROR][Trainer::train]")
            print("\t GPU memory not enough!")
            return False
        print("[INFO][Trainer::train]")
        print("\tTF GPU memory used: " + str(allowable_fraction*100) + "%")

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=allowable_fraction)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            writer = tf.summary.FileWriter(f'{experiment_dir}/log', session.graph)
            print('Initializing variables...')
            session.run([init_op])

        if FLAGS.visualize:
          visualize_data(session, model_config.inputs['dataset'])

        # Check whether the checkpoint directory already exists (resuming) or
        # needs to be created (new model).
        if not os.path.isdir(checkpoint_dir):
          print('No previous checkpoint detected, training from scratch.')
          os.makedirs(checkpoint_dir)
          # Serialize hparams so eval can load them:
          hparam_path = f'{checkpoint_dir}/hparam_pickle.txt'
          if not file_util.exists(hparam_path):
            hparams.write_hparams(model_config.hparams, hparam_path)
          initial_index = 0
        else:
          print(
              f'Checkpoint root {checkpoint_dir} exists, attempting to resume.')
          latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
          print(f'Latest checkpoint: {latest_checkpoint}')
          saver.restore(session, latest_checkpoint)
          initial_index = session.run(global_step_op)
          print(f'The global step is {initial_index}')
          initial_index = int(initial_index)
          print(f'Parsed to {initial_index}')
        start_time = time.time()
        log_every = 10
        for i in range(initial_index, FLAGS.train_step_count):
          print(f'Starting step {i}...')
          is_summary_step = i % FLAGS.summary_step_interval == 0
          if is_summary_step:
            _, summaries, loss = session.run(
                [model_config.train_op, summary_op, model_config.loss])
            writer.add_summary(summaries, i)
          else:
            _, loss = session.run([model_config.train_op, model_config.loss])
          if not (i % log_every):
            end_time = time.time()
            steps_per_second = float(log_every) / (end_time - start_time)
            start_time = end_time
            print(f'Step: {i}\tLoss: {loss}\tSteps/second: {steps_per_second}')

          is_checkpoint_step = i % FLAGS.checkpoint_interval == 0
          if is_checkpoint_step or i == FLAGS.train_step_count - 1:
            ckpt_path = os.path.join(checkpoint_dir, 'model.ckpt')
            print(f'Writing checkpoint to {ckpt_path}...')
            saver.save(session, ckpt_path, global_step=i)
        print('Done training!')
        return True

def demo():
    trainer = Trainer()
    trainer.train()
    return True

if __name__ == "__main__":
    demo()

