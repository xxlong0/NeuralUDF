# NeuralUDF: Learning Unsigned Distance Fields for Multi-view Reconstruction of Surfaces with Arbitrary Topologies (CVPR2023)

## [Project Page](https://www.xxlong.site/NeuralUDF/) | [Paper](https://arxiv.org/abs/2211.14173) 

We have released the core codes now and will gradually release all the codes in the following days.

![](./docs/images/teaser.png)

## Introduction
We present a novel method, called NeuralUDF, for reconstructing surfaces with arbitrary topologies from 2D images via volume rendering.
However, these methods are limited to objects with closed surfaces since they adopt Signed Distance Function (SDF)
as surface representation which requires the target shape to be divided into inside and outside.
In this paper, we propose to represent surfaces as the Unsigned Distance Function (UDF) and
develop a new volume rendering scheme to learn the neural UDF representation.
Specifically, a new density function that correlates the property of UDF with the volume rendering scheme is introduced for robust optimization of the UDF fields.
Experiments on the DTU and DeepFashion3D datasets show that our method not only enables high-quality reconstruction of non-closed shapes with complex typologies, but also achieves comparable performance to the SDF based methods on the reconstruction of closed surfaces.
        
        
## Usage
            
### Setup environment
Set up a conda environment with the right packages using:
```
conda env create -f conda_env.yml
conda activate neuraludf
```

We leverage [MeshUDF](https://github.com/cvlab-epfl/MeshUDF) to extract mesh from the learned UDF field. 
Thank them for the great work.
To compile the custom version for your system, please run:
```
cd custom_mc
python setup.py build_ext --inplace
cd ..
```



#### Data Convention
The DTU data and Deepfashion3d data are organized as follows:

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
```

Here the `cameras_xxx.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), 
where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

### Running

- **On objects with closed surfaces (DTU)**

The training has two stages. 
We apply blending-based patch loss (used in SparseNeuS) to further improve the reconstruction.

```shell
bash bashs/bash_dtu_blending.sh --gpu 0 --case scan118
bash bashs/bash_dtu_blending_ft.sh --gpu 0 --case scan118
```

- **On objects with open surfaces (Deepfashion3D)**
```shell

```

- **Extract surface from trained model** 

```shell
python exp_runner_blending.py --mode validate_udf_mesh --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.


### Train NeuralUDF with your custom data

More information can be found in [preprocess_custom_data](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data) of NeuS.

### Discussions and future work
As we stated in the paper, it's more difficult to train a UDF field than a SDF field, 
since UDF doesn't adopt any geometric assumption(like the surfaces are closed) and UDF is not differentiable at zero-level sets.
Although we propose a series of strategies to alleviate the problem,
there are still some limitations, and hope that they can be improved in the future.
-  The weight of the geometric regularization sometimes is sensitive to some cases, and need to be tuned for better results.
Maybe a more robust regularization can handle this.
- How to initialize the UDF field for open surfaces ? In the work, we still adopt sphere initialization.
- How to extract mesh from the optimized UDF in a more robust way ? MeshUDF is a great work, 
but it's sensitive to the gradients near zero-level sets, and cannot handle non-manifold surfaces.

## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{long2022neuraludf,
  title={NeuralUDF: Learning Unsigned Distance Fields for Multi-view Reconstruction of Surfaces with Arbitrary Topologies},
  author={Long, Xiaoxiao and Lin, Cheng and Liu, Lingjie and Liu, Yuan and Wang, Peng and Theobalt, Christian and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2211.14173},
  year={2022}
}
```

## Acknowledgement

Some code snippets are borrowed from [NeuS](https://github.com/Totoro97/NeuS), 
[MeshUDF](https://github.com/cvlab-epfl/MeshUDF) and [SparseNeuS](https://github.com/xxlong0/SparseNeuS). 
Thanks for these great projects.