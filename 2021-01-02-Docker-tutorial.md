---
title: Docker tutorial
tags: [docker]
style: fill
color: warning
description: ESRGAN implemetation using keras. An enhanced version of the SRGAN by modifying the model architecture and loss functions.
---

1.0 Run docker container
`docker container run hello-world`

1.1 Docker images
* Pull image
The `pull` command fetches the alpine image from the Docker registry and saves it in our system.
`docker image pull alpine`

* List docker images
`docker image ls`

Docker container functions at aplication layer so they skip most of the steps VMs require and just run what is required for the app. A VM has to emulate a full hardware stack, boot an operating system, and then launch your app - it’s a virtualized hardware environment.

* List docker running containers
`docker container ls`

* List all docker containers
`docker container ls -a`

1.2 Container isolation
Why are there so many containers listed if they are all from the alpine image?

This is a critical security concept in the world of Docker containers! Even though each docker container run command used the same alpine image, each execution was a separate, isolated container. Each container has a separate filesystem and runs in a different namespace; by default a container has no way of interacting with other containers, even those from the same image.

* Interactive shell in container
`docker container run -it alpine /bin/ash`

* Start specific container using container ID
`docker container start <container ID>`

* Execute command in running container
`docker container exec <container ID> ls`


* Images - The file system and configuration of our application which are used to create containers. To find out more about a Docker image, run `docker image inspect alpine`. In the demo above, you used the `docker image pull` command to download the alpine image. When you executed the command `docker container run hello-world`, it also did a `docker image pull` behind the scenes to download the hello-world image.
* Containers - Running instances of Docker images — containers run the actual applications. A container includes an application and all of its dependencies. It shares the kernel with other containers, and runs as an isolated process in user space on the host OS. You created a container using `docker run` which you did using the alpine image that you downloaded. A list of running containers can be seen using the `docker container ls` command.
* Docker daemon - The background service running on the host that manages building, running and distributing Docker containers.
* Docker client - The command line tool that allows the user to interact with the Docker daemon.
* Docker Hub - Store is, among other things, a registry of Docker images. You can think of the registry as a directory of all available Docker images. You’ll be using this later in this tutorial.


* Inspect all the changes made to conatainer 
`docker container diff <container ID>`

* commit container to create image
`docker container commit CONTAINER_ID`

* Tag docker image
`docker image tag <IMAGE_ID> <IMAGE_TAG>`

2.1 Image creation using a Dockerfile

