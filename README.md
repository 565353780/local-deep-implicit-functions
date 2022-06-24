# Local Deep Implicit Functions

## Source

```bash
https://github.com/google/ldif
```

## Install

```bash
sudo apt install mesa-common-dev libglu1-mesa-dev libosmesa6-dev libxi-dev libgl1-mesa-dev libglew-dev
sudo apt install --reinstall libgl1-mesa-glx

conda create -n ldif python=3.7
conda activate ldif

pip install trimesh tqdm absl_py matplotlib numpy parameterized six scikit-image scipy \
      tensorflow-hub joblib pandas tabulate apache-beam pillow tensorboard
pip install tensorflow-gpu==1.15
pip install protobuf==3.20
```

## Build

```bash
cd ldif/gaps
make mesa -j

cd ../..
./ldif/ldif2mesh/build.sh
```

## Dataset

## Tool Usage

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
./train.sh
```

```
tensorboard --logdir ./trained_models/sif-transcoder-<experiment_name>/log
```

## Eval

```
./eval.sh
```

## Interactive Sessions

see

```bash
ldif_example_inference.ipynb
```

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

