# 3DGStream

Official repository for the paper "3DGStream: On-the-fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos".

> **3DGStream: On-the-fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos**  
> [Jiakai Sun](https://sjojok.github.io), Han Jiao, [Guangyuan Li](https://guangyuankk.github.io/), Zhanjie Zhang, Lei Zhao, Wei Xing  
> *CVPR 2024 __Highlight__*  
> [Project](https://sjojok.github.io/3dgstream)
| [Paper](https://arxiv.org/pdf/2403.01444.pdf)
| [Suppl.](https://drive.google.com/file/d/18x3oNsFa3UtG1WqndeKLTjEMluxyRBJ6/view)
| [Bibtex](##Bibtex)
| [Viewer](https://github.com/SJoJoK/3DGStreamViewer)



## Release Roadmap

- [x] Open-source [3DGStream Viewer](https://github.com/SJoJoK/3DGStreamViewer)

    - [x] Free-Viewpoint Video
      
- [x] Unorganized code with few instructions (around May 2024)

    - [x] Pre-Release
      
- [ ] Refactored code with added comments (after CVPR 2024)

- [ ] 3DGStream v2 (hopefully in 2025)

## Step-by-step Tutorial for 3DGStream (May Ver.)

1. Follow the instructions in [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) to setup the environment and submodules, after that, you need to install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

   You can use the same Python environment configured for gaussian-splatting. However, it is necessary to install tiny-cuda-nn and reinstall the submodules/diff-gaussian-rasterization by running `pip install submodules/diff-gaussian-rasterization`. Additionally, we recommend using PyTorch version 2.0 or higher for enhanced performance, as we utilize `torch.compile`. If you are using a PyTorch version lower than 2.0, you may need to comment out the lines of the code where `torch.compile` is used.


2. Follow the instructions in [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) to create your COLMAP dataset based on the images of the timestep 0 , which will end-up like: 

   ```
   <frame000000>
   |---images
   |   |---<image 0>
   |   |---<image 1>
   |   |---...
   |---distorted
   |	|---sparse
   |       |---0
   |           |---cameras.bin
   |           |---images.bin
   |           |---points3D.bin
   |---sparse
       |---0
           |---cameras.bin
           |---images.bin
           |---points3D.bin
   ```

   You can use *test/flame_steak_suite/frame000000* for experiment on the `flame steak` scene.

3. Follow the instructions in [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) to get a **high-quality** init_3dgs (sh_degree = 1, i.e., train with `--sh_degree 1`) from the above colmap dataset, which will end-up like:

   ```
      <init_3dgs_dir>
      |---point_cloud
      |   |---iteration_7000
      |   |   |---point_cloud.ply
      |   |---iteration_15000
      |   |---...
      |---...   
   ```

   You can use *test/flame_steak_suite/flame_steak_init* for experiment on the `flame steak` scene. 

   Since the training of 3DGStream is orthogonal to that of init_3dgs, you are free to use any method that enhances the quality of init_3dgs, provided that the resulting ply file remains compatible with the original [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

4. Prepare the multi-view video dataset:

   1. Extract the frames and organize them like this:

      ```
      <scene>
      |---frame000001
      |   |---<image 0>
      |   |---<image 1>
      |   |---...
      |---frame000002  
      |---...
      |---frame000299
      ```

      If you intend to use the data we have prepared in the test/flame_steak_suite, ensure that the images are named following the pattern `cam00.png`, ..., `cam20.png`. This is necessary because COLMAP references images by their file names.

      For convenience, we assume that you extract the frames of the `flame steak` scene into *dataset/flame_steak*. This means your folder structure should look like this:

      ```
      dataset/flame_steak
      |---frame000001
      |   |---cam00.png
      |   |---cam01.png
      |   |---...
      |---frame000002  
      |---...
      |---frame000299
      ```     

   2. Copy the camera infos by `python scripts/copy_cams.py --source <frame000000> --scene <scene>`:

       ```
      <scene>
      |---frame000001
      |   |---sparse
      |   |   |---...
      |   |---<image 0>
      |   |---<image 1>
      |   |---...
      |---frame000002  
      |---frame000299
      |---distorted
      |   |---...
      |---...      
       ```

      You can run

      ```bash
      python scripts/copy_cams.py --source test/flame_steak_suite/frame000000 --scene dataset/flame_steak`
      ```

      to prepare for conducting experiment on the `flame steak` scene.

   4. Undistort the images by `python convert_frames.py -s <scene> --resize`, then the dataset will end-up like this:

      ```
      <scene>
      |---frame000001
      |   |---sparse
      |   |---images
      |       |---<undistorted image 0>
      |       |---<undistorted image 1>
      |       |---....
      |   |---<image 0>
      |   |---<image 1>
      |   |---...
      |---frame000002  
      |---...
      |---frame000299
      ```

      You can run

      ```bash
      python convert_frames.py --scene dataset/flame_steak --resize
      ```

      to prepare for conducting experiment on the `flame steak` scene.

5. Warm-up the NTC

   Please refer to the *scripts/cache_warmup.ipynb* notebook to perform a warm-up of the NTC.

   For better performance, it's crucial to define the corners of the Axis-Aligned Bounding Box that approximately enclose your scene. For instance, in a scene like `flame salmon`, the AABB should encompass the room while excluding any external landscape elements. To set the coordinates of the AABB corners, you should directly hard-code them into the `get_xyz_bound` function.


6. GO!

   Everything is set up, just run

   ```bash
   python train_frames.py --read_config --config_path <config_path> -o <output_dir> -m <init_3dgs_dir>  -v <scene> --image <images_dir> --first_load_iteration <first_load_iteration>
   ```

   Parameter explanations:
   * `<config_path>`: We provide a configuration file containing all necessary parameters, available at *test/flame_steak_suite/cfg_args.json*.
   * `<init_3dgs_dir>`: Please refer to the section 2 of this guidance.
   * `<scene>`: Please refer to the section 4.2 of this guidance.
   * `<images_dir>`: Typically named `images`, `images_2`, or `images_4`. 3DGStream will use the images located at *\<scene\>/\<frame[id]\>/\<images_dir\>* as input.
   * `<first_load_iteration>`: 3DGStream will initialize the 3DGS using the point cloud at *\<init_3dgs_dir\>/\<point_cloud\>/iteration_\<first_load_iteration\>/point_cloud.ply*.
   * Use `--eval` when you have a test/train split. You may need to review and modify `readColmapSceneInfo` in *scene/dataset_renders.py* accordingly.
   * Specify `--resolution` only when necessary, as reading and resizing large images is time-consuming. Consider resizing the images before 3DGStream processes them.
   * About NTC:
      - `--ntc_conf_path`: Set this to the path of the NTC configuration file (see *scripts/cache_warmup.ipynb*, *configs/cache/* and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)).
      - `--ntc_path`: Set this to the path of the pre-warmed parameters (see *scripts/cache_warmup.ipynb*).

   You can run

   ```bash
   python train_frames.py --read_config --config_path test/flame_steak_suite/cfg_args.json -o output/Code-Release -m test/flame_steak_suite/flame_steak_init/ -v <scene> --image images_2 --first_load_iteration 15000 --quiet
   ```

   to conduct the experiments on the `flame steak` scene.

8. Evaluate Performance

   * PSNR: Average among all test images

   * Per-frame Storage:  Average among all frames (including the first frame)

     For a multi-view videos that has 300 frames, the per-frame storage is $$\frac{(\text{init3dgs})+299*(\text{NTC}+\text{new3dgs})}{300}$$​

   * Per-frame Training Time: Average among all frames  (including the first frame)

   * Rendering Speed

     There are serval ways to evaluate the rendering speed: 

     * **[SIBR-Viewer](https://gitlab.inria.fr/sibr/sibr_core)** (As presented in our paper)
      
       Integrate 3DGStream into SIBR-Viewer for an accurate measurement. If integration is too complex, approximate the rendering speed by:

       1. Use the SIBR-Viewer to render the init_3dgs and get the rendering speed

       2. Profiling `query_ntc_eval` using *scripts/cache_profile.ipynb*.

       3. Summing the measurements for an estimated total rendering speed, like this:
          
            | Step             | Overhead(ms) | FPS  |
            | ---------------- | ------------ | ---- |
            | Render w/o NTC   | 2.56         | 390  |
            | + Query NTC      | 0.62         |      |
            | + Transformation | 0.02         |      |
            | + SH Rotation    | 1.46         |      |
            | Total            | 4.46         | 215  |

          To isolate the overhead for each process, you can comment out the other parts of the code.

     * **[3DGStreamViewer](https://github.com/SJoJoK/3DGStreamViewer)**

       You can use *scripts/extract_fvv.py* to re-arrange the output of 3DGStream and render it with 3DGStreamViewer

     * **Custom Script**

       Write a script that loads all NTCs and additional_3dgs and renders the test image for every frame. For guidance, you can look at the implementation within [3DGStreamViewer](https://github.com/SJoJoK/3DGStreamViewer)

## Acknowledgments

We acknowledge the foundational work of [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), which form the basis of the 3DGStream code. Special thanks to [Qiankun Gao](https://github.com/gqk) for his feedback on the pre-release version.

## Bibtex
```
@article{sun20243dgstream,
  title={3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos},
  author={Sun, Jiakai and Jiao, Han and Li, Guangyuan and Zhang, Zhanjie and Zhao, Lei and Xing, Wei},
  journal={arXiv preprint arXiv:2403.01444},
  year={2024}
}
```
