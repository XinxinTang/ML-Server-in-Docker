# Multi-model unified system


## Description

Environments: 
  Flask==0.10.1;
  pandas==0.22.0;
  tensorflow==1.6.0;
  rpy2==2.9.4;
  tzlocal

1 Two machine learning models, one is built by Python with tensorflow, the other is built by R.

2 Flask loads these two models as a predictive service so that end-user can get a response from the target Machine Learning Model.

![framework](https://github.com/XinxinTang/ML-Server-in-Docker-/blob/master/pics/Screen%20Shot%202018-11-24%20at%206.35.59%20PM.png)

## Installation

1 Install and open Docker

2 Run commands in the following:
```
docker build -t Multi-model_Platform:1.5
docker run -d -p 5000:5000 Multi-model_Platform:1.5
```

3 Send request from end-user

```
curl -X POST --header "Content-Type:application/json" -d '{"SepalLength": [5.1, 5.9],
"SepalWidth": [3.3, 3.0], "PetalLength": [1.7, 4.2], "PetalWidth": [0.5, 1.5]}'
0.0.0.0:5000/tensorflow-iris/predict

curl -X POST --header "Content-Type:application/json" -d '{"SepalLength": [5.1, 5.9],
"SepalWidth": [3.3, 3.0], "PetalLength": [1.7, 4.2], "PetalWidth": [0.5, 1.5]}' 0.0.0.0:5000/r-iris/predict
```
![Output](https://github.com/XinxinTang/ML-Server-in-Docker-/blob/master/pics/Screen%20Shot%202018-11-24%20at%206.37.33%20PM.png)
