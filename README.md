

### Data preprocessing

Before performing the editing tasks below, we should first pre-process the data via

```shell
python -m face_process.detail_process --input ./demo/input --output ./demo/output --gpu_ids -1

```


### Age progression animation

```shell
python -m experiments both_cond age_progression SEMM --gpu_id 0
```

The results will be in `./demo/output/${filename}/age_progression/`.

### Blendshape animation

```shell
python -m experiments both_cond bs_anime SEMM --gpu_id 0
```

The results will be in `./demo/output/${filename}/bs_anime_${clip_name}/`.

### Interactive wrinkle line editing

```shell
python -m experiments both_cond demo SEMM --gpu_id 0
```

An GUI window will pop up. You can import a displacement map and edit it by drawing or erasing lines.

<details>
    <summary><b>Common issues</b></summary>
If you encounter the problem "qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "${conda_path}/envs/semm/lib/python3.7/site-packages/cv2/qt/plugins" even though it was found", that's a conflict between opencv-python and pyqt5. Consider using
    
```shell
conda install -c anaconda py-opencv
conda install -c alges pyqt 
```
</details>

## Training

We provide a small sample dataset that comprises identities in FaceScape's [publishable_list_v1.txt](https://facescape.nju.edu.cn/static/publishable_list_v1.txt). You can commence training by utilizing the sample dataset through the following command:

```shell
python -m experiments both_cond train SEMM --gpu_id 01
```
The default configuration assumes running on two RTX 3090 GPUs.

### Training data preprocessing

Given that the sample dataset is insufficiently small to lead to favorable results, it is necessary to apply and download `Topologically Uniformed Model(TU-Model)` (`facescape_trainset_001_100.zip` ~ `facescape_trainset_801_847.zip`) in the FaceScape dataset. 

Once downloaded, extract the dataset to `/path/to/FaceScape`. The folder should comprise of 847 folders, labeled from `1` to `847`.

Next, use the following command to process the training dataset:

```shell
python -m detail_shape_process.train_detail_process --input /path/to/FaceScape --output /path/to/processed/dataset
```

Lastly, make sure to update the subsequent line:

```python
        opt.set(
            dataroot="./predef/sample_dataset/",  # just a small sample dataset
```

in `experiments/both_cond_launcher.py` to point to the processed dataset.


}
```

