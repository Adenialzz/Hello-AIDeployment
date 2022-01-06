mkdir model_file && cd model_file
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg?raw=true -O model_file/yolov3-tiny.cfg
