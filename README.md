# robbenblick
A Computer Vision project


## Scripts
There are three major scripts, create_dataset.py, yolo.py and run_fiftyone.

### `create_dataset.py`
Prepares and formats raw image and annotation data for training. Handles data extraction, organization, and conversion to required formats for using ultralytics YOLO models. Parameters can be set in the [configs/create_dataset.yaml](configs/create_dataset.yaml) file.

Example:

```
python -m robbenblick.create_dataset
```

### `yolo.py`
Trains and evaluates YOLO-based object detection models. Includes configuration options for model architecture, training parameters, and evaluation metrics. Parameters can be set in the [configs/create_dataset.yaml](configs/model.yaml)

Example:

```
python -m robbenblick.yolo --mode train --run_id name-of-run

python -m robbenblick.yolo --mode predict --run_id name-of-run
```

The ```--mode predict``` will save the predictions in folder ```run/detect/name-of-run_predict```


### `run_fiftyone.py`
Launches FiftyOne for dataset visualization and analysis.

Example:
```
python -m robbenblick.run_fiftyone --run_id name-of-run --dataset cvat --recreate
python -m robbenblick.run_fiftyone --run_id name-of-run --dataset yolo --recreate
```

The ```--recreate``` flag would delete any current fo datasets with the same name and create a new one.
If not set the old dataset will not be overwritten.

## Environment setup

```conda env create --file environment.yml```

```conda activate RobbenBlick```

To run the pre-commit hooks:

```pre-commit run```



## CVAT
The annotation is done in CVAT.

To get the data locally to train the model:
1. Login to CVAT and go to jobs
2. Export job as dataset ```CVAT for images```, toggle on "save images"
3. Download when finished

Unsip the data and place data in paths ```data/raw/images``` and ```data/raw/annotations.xml```.


>[!NOTE]
>.TIF images can not be displayed in all browsers without installed extensions. If you see "image failed to load", try another browser.


# Fiftyone
Export the dataset from fiftyone and unsip it.
Put it at the local path:
```
data/raw/images
data/raw/annotations.xml
```

Run fiftyone:
```
python -m robbenblick.run_fiftyone --dataset cvat --recreate
```

# Developent

# Known issues

## FiftyOne
### failed to bind port
If you get the error
```fiftyone.core.service.ServiceListenTimeout: fiftyone.core.service.DatabaseService failed to bind to port```
Try to kill the sessions with
```
pkill -f fiftyone
pkill -f mongod
```
and try again.
