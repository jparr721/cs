# Visual Magic
Jarred Parr, Thomas Bailey, Brendan Caywood, Alexander Fountain

This project needs the following datasets:
```
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
$ wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
```
These datasets are massive, so be ready for that.

### Running
Just make a virtual environment and install via `pip install -r requirements.txt` and then run the runner file `./runner.py` and watch the magic.

You may need to update the path directories inside of `runner.py` depending on how you downloaded the datasets.
