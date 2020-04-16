# visiter-localization

Track and predict basic info about the visitors using rgbd camera.

Requires:
- opencv_dnn (better if opencv is compiled with dnn_cuda)
- [`pymynt`](https://github.com/tccoin/pymynt) for capture rgbd image with MYNT depth camera.
- `agcnn/` weights for [Age and Gender Classification Using Convolutional Neural Networks](https://talhassner.github.io/home/publication/2015_CVPR)
  - `age_net.caffemodel`
  - `deploy_age.prototxt`
  - `deploy_gender.prototxt`
  - `gender_net.caffemodel`
  - `mean.binaryproto`
- `haarcascade/haarcascade_frontalface_alt.xml` from [opencv/data/haarcascades/](https://github.com/opencv/opencv/tree/master/data/haarcascades): face recognization
- [`YOLO` homepage](https://pjreddie.com/darknet/yolo/): Real-Time Object Detection
