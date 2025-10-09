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
python run_fiftyone.py
```

