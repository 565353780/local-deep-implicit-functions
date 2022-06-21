# Local Deep Implicit Functions

## Source

```bash
https://github.com/google/ldif
```

## Install

```bash
sudo apt install mesa-common-dev libglu1-mesa-dev libosmesa6-dev libxi-dev libgl1-mesa-dev libglew-dev
sudo apt install --reinstall libgl1-mesa-glx

conda env create -n ldif python=3.8
conda activate ldif

pip install trimesh tqdm absl_py matplotlib numpy parameterized tensorboard
```

## Build

```bash
cd ldif/gaps
make mesa -j

cd ../..
./ldif/ldif2mesh/build.sh
```

## Dataset

```bash
./ldif/gaps/bin/x86_64/msh2msh mesh.obj mesh.ply
```

or

```bash
./ldif/gaps/bin/x86_64/msh2df input.ply tmp.grd -estimate_sign -spacing 0.002 -v
./ldif/gaps/bin/x86_64/grd2msh tmp.grd output.ply
rm tmp.grd
```

then, save as

```bash
<path-to-root>/{train/val/test}/{class names}/{.ply files}
```

```
python meshes2dataset.py --mesh_directory <path-to-dataset_root> \
  --dataset_directory <path-to-nonexistent_output_directory>
```

## Train

```
python train.py --dataset_directory <path-to-dataset_root> \
  --experiment_name [name] --model_type {ldif, sif, or sif++}
```

```
tensorboard --logdir <ldif_root>/trained_models/sif-transcoder-<experiment_name>/log
```

Warning: Training an LDIF from scratch takes a long time. SIF also takes a while, though
not nearly as long. The expected performance with a V100 and a batch size of 24 is
3.5 steps per second for LDIF, 6 steps per second for SIF. LDIF takes about 3.5M steps
to fully converge on ShapeNet, while SIF takes about 700K. So that is about 10 days to
train an LDIF from scratch, and about 32 hours for SIF. Note that LDIF performance is
pretty reasonable after 3-4 days, so depending on your uses it may not be necessary to
wait the whole time. The plan is to 1) add pretrained checkpoints (the most pressing
TODO) and 2) add multi-gpu support, later on, to help mitigate this issue. Another
practical option might be switching out the encoder for a smaller one, because most
of the training time is the forward+backward pass on the ResNet50.

## Evaluation and Inference

To evaluate a fully trained LDIF or SIF network, run the following:

```
python eval.py --dataset_directory [path/to/dataset_root] \
  --experiment_name [name] --split {test, val, or train}
```

This will compute metrics over the dataset and then print out the result to the
terminal. By default, it will print out a table of results and LaTeX code. In
addition, there are flags `--save_results`, `--save_meshes`, and `--save_ldifs`,
which can be set to true. If they are set, the code will also write
1) pandas-readable CSV files containing the metrics for each mesh and class,
2) a directory of result meshes generated by the algorithm, and/or
3) a directory of txt files containing the actual LDIF/SIF representation
(parseable by qview, ldif2mesh, and the Decoder class in the ipynb). If these
flags are set, then `--result_directory` must also be provided indicating where
they should be written.

## Interactive Sessions

You can run the code interactively in an Jupyter notebook. To do so, open the
provided file `ldif_example_inference.ipynb` with Jupyter and attach
it to a python 3.6 kernel with the requirements.txt installed. Then follow the
prompts in the notebook. The notebook has a demo of loading a mesh, creating an
example, running inference, visualizing the underlying SIF elements, extracting
a mesh, and computing metrics. There is additional documentation in the .ipynb.

## Unit Tests

There is a script `unit_test.sh`. If you want to check whether the code is
installed correctly and works, run it with no arguments. It will make a small
dataset using open source models and train/evaluate an LDIF network. Note that
it doesn't train to convergence so that it doesn't take very long to run. As a
result, the final outputs don't look very good. You can fix this by setting the
step count in `unit_test.sh` higher (around 50K steps is definitely sufficient).

The code also has some unit tests for various pieces of functionality. To run a
test, cd into the directory of the `*_test.py` file, and run it with no arguments.
Please be aware that not all of the code is tested, and that the unit tests
aren't well documented. The easiest way to check if the code still works is by
running `./unit_test.sh`.

## Other code and PyTorch

In addition to the scripts described above, there are also model definitions and
beam pipelines provided for generating datasets and running inference on a
larger scale. To use these scripts, it would be necessary to hook up your own
beam backend.

There is also very limited PyTorch support in the `ldif/torch` directory. This code
is a basic implementation of SIF that can't train new SIF models, but can load
and evaluate SIFs generated by the tensorflow training+evaluation code. It is mainly
intended for using SIF correspondences as a building block of another unrelated
project in PyTorch. Note that PyTorch is not included in the `requirements.txt`, and
the `torch/` subdirectory is independent from the rest of the code base (it interacts
only through the `.txt` files written by the tensorflow code and takes no dependencies
on the rest of this codebase). To use it, it is probably easiest to just download
the `torch/` folder and import the `sif.py` file as a module.

## Updates to the code

* The code now supports tfrecords dataset generation and usage. This reduces
  the IO workload done during training. It is enabled by default. Existing
  users can git pull, rerun `meshes2dataset.py` with `--optimize` and 
  `--optimize_only`, and then resume training where they left off with the
  new dataset improvements. If you currently experience less than 100% GPU
  utilization, it is highly recommended. Note it increases dataset size by
  3mb per example (and can be disabled with `--nooptimize`).
  
* Support for the inference kernel on Volta, Turing and CC 6.0 Pascal cards
  should now work as intended. If you had trouble with the inference kernel,
  please git pull and rerun `./build_kernel.sh`.
  

## TODOS

This is a preliminary release of the code, and there are a few steps left:

* Pretrained model checkpoints are on the way. In the mean-time, please see
`reproduce_shapenet_autoencoder.sh` for shapenet results.
* This code base does not yet support training a single-view network. In the
  mean-time, the single-view network architecture has been provided (see
  `ldif/model/hparams.py` for additional information).
* While the eval code is fast enough for shapenet, the post-kernel
eval code is written in numpy and is an unnecessary bottleneck. So inference at
  256^3 takes a few seconds per mesh, even though the kernel completes in
  ~300ms.
* Pressing the 'f' key in a qview visualization session will extract a mesh
and show it alongside the SIF elements; however it only considers the analytic
parameters. Therefore, it shows a confusing result for LDIF representations,
  which also have neural features.
* To make setup easier, we would like to provide a docker container
that is ready to go.
