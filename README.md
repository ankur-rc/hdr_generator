# HDR Generator
Generate HDR images. Based on Debevec, et. al. 1997.

```
Generate HDR images from SDR images taken with different exposures
Filenames should follow x_y_z.jpg format, where 'x' is name, 'y/z' is the exposure time.
            
    Usage: ./hdr dirpath calibdir calibflag lambda alpha
            dirpath:    Path to images
            calibdir:   Path to load/save calibration files
            calibflag:  Flag to specify if we want to calibrate or not
            lambda:     Regularization constant while calibrating
            alpha:      Global tone-mapping constant
            opencv_cmp: Compare with openCV's Debevec-Durand Algorithm
```
## Compile
Run
```
$ mkdir <build_dir> && cd <build_cir> && cmake ..
```

To configure, run (from build directory):
```
$ ccmake .
```

## Depends on:
- Eigen 3.3
- Boost 1.69.0
- Opencv 3.3.1
- PythonLibs 2.7
