# Robust Wireless Sensing using Probabilistic and Statistical Assessments ：A Interactive Demo

We provide a [Docker Image](#docker) to support artifact for our paper (RISE) on MobiCom 2021 paper on Robust Wireless Sensing. 

Our docker image contains minimal working examples which can be evaluated in a reasonable amount of time. 


# Step-by-Step Instructions <br id = "docker">
*Disclaim:
Although we have worked hard to ensure our demo are robust, our tool remains a *research prototype*. It can still have glitches when using in complex, real-life settings. If you discover any bugs, please raise an issue, describing how you ran the program and what problem you encountered. We will get back to you ASAP. Thank you.*

## ★ Docker Image <br id = "dockerimg">

We prepare our demo within a Docker image to run "out of the box". A reduced-sized Docker image can be downloaded from [here](link：https://pan.baidu.com/s/1_K5XTQtzNBxq9qFo8kBK1Qpassword：vf8h). 

Our docker image was tested on a host machine running Ubuntu 18.04 and Windows 10.  

### 1.  Load the Docker Image 
After downloading the [docker image](#dockerimg), using the following commands to load the docker image (~30 minutes on a laptop for the reduced sized image) on the host machine:

- ###### Unpack the docker

`unzip RISE_docker.zip`  

- ###### Import Docker image

`cat RISE_docker.rar | docker import - rise:202108`

- ###### Run Docker

`docker run -itd --name rise -p 3921:3921 rise:202108 /bin/bash`

- ###### Enter the docker

`docker exec -it rise /bin/bash`

- ###### Enter RISE code

`cd root/RISE-Version2`

`Password: nisl8830`

### 2. Evaluation  

Notes for how to reuse the demo can be found at [here](https://github.com/jiaojiao1234/RISE/blob/master/Jupyter/Main3.ipynb)



