# Vehicle Counting with DeepSort and YOLOv5

The objective of this project is to `detect` and `track` the number of vehicles in a predefined video file `toycars.mp4`.  
The output file should include a counter of the number of vehicles in each frame, as well as a running count of the number of vehicles seen so far.  

For the assignment requirement, the video file used will be the first 15 secs of [Different types of vehicles Moving on On the roof](https://www.youtube.com/watch?v=ucEG-uM5v_0)

![Different types of vehicles Moving on On the roof](data/gifs/original.gif)

To implement the `detection` part of the problem, we will be using [YOLOv5](https://github.com/ultralytics/yolov5).  

The `tracking` portion will be implemented using the [DeepSORT](https://github.com/nwojke/deep_sort)  algorithm.  

To save the *best image* of each vehicle detected, a measure of the euclidean distance from the bounding box center to the frame center point is also implemented. The idea is that the best image to save should be located closest to the frame center.  

### Final Output

![Final Output](data/gifs/final_output.gif)


## Installation  

Clone the repository using


```shell
git clone https://github.com/weishengtoh/computervision_assignment.git
```

Once that is completed, navigate to the folder and create the conda environment using

```shell
cd computervision_assignment
```

```shell
conda env create -f environment.yml
```

Activate the conda environment that was created using

```shell
conda activate cv_assignment
```


## Usage  

There are two methods to edit the model configuration prior to execution:
1. By modifying the default behaviour using config file
2. Using hydra to override the config during execution

By default, the video output will be saved in the folder `data/output` and the images of the vehicles will be saved in `data/images`. 

### Modifying the default behaviour using config file

The default configuration is stored in the yaml file `configs` in the folder `configs`.  

Modifying the values in the config file will change the default behaviour when the main file is executed using the command

```shell
python main.py
```

### Using hydra to override the config during execution

The configuration defined in the config file may also be overridden by hydra during execution.  
To override the parameters without changing the default behaviour, they have to be defined in the command during execution.  

For example, to change the verbosity of the outputs and the labels that are detected by YOLOv5, the command to run should be:  

```shell
python main.py "main.verbose=True" "parameters.labels=[car]"
```


## Next Steps
- The appearance feature extractor used in DeepSORT was trained using a CNN archietecture on 1,100,000 images of 1,261 pedestrains, which might not be suitable for extracting features of vehicles. It might be more appropriate to retrain the feature extractor on a new dataset.
- Only YOLOv5 was used for the object detection. Might wish to explore using other models


## References
**Video File**
- [Different types of vehicles Moving on On the roof](https://www.youtube.com/watch?v=ucEG-uM5v_0)

**DeepSORT**  
- [Repository](https://github.com/nwojke/deep_sort)  
- [Paper](https://arxiv.org/abs/1703.07402)  

**YOLOv5**
- [Repository](https://github.com/ultralytics/yolov5)
