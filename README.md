# robbenblick
A Computer vision project


# Environment setup 

```conda env create --file environment.yml```
```conda activate RobbenBlick```

# CVAT 
The annotation is done in CVAT. 

>[!NOTE]
>.TIF images can not be displayed in all browsers without installed extensions. If you see "image failed to load", try another browser. 


# Fiftyone 
Export the dataset from fiftyone and unsip it. 
Put it at the local path: 
```
data/robben/images 
data/robben/annotations.xml 
```
Run fiftyone: 
```
python -m robbenblick.run_fiftyone --dataset cvat --recreate
```

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