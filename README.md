[![License](https://img.shields.io/badge/License-MIT-blue.svg)](/LICENSE)

# Triangle Counting
This project provides a program which can count the Triangles in undirected graph.

## Background
[基于GPU服务器的图数据三角形计数算法设计与性能优化](https://www.datafountain.cn/competitions/321/details/rule)

## Environment
- GPU: NVIDIA® Tesla® P100
- CPU: Intel® Xeon® Processor E5-2682 v4
- RAM: 60 GiB
- SSD: 440 GiB
- OS: ubuntu 16.04
- compiler: g++ 5.4.0
- open-mpi version: 201307
- CUDA Driver Version: 9.0
- CUDA Runtime Version: 8.0

## How to Use
Build the project with:
```
make
```
If your datafile is "/data/s27.kron.edgelist", then just run
```
./main
```
or else run
```
./main -f some_url
```
while some_url is your datafile

## Answer and performance

- soc-LiveJournal1.bin

  Triangle number = 285730264

  Time = 6(s)

- s24.kron.edgelist

  Triangle number = 10286638314

  Time = 28(s)

- s26.kron.edgelist

  Triangle number = 49167172995

  Time = 132(s)

- s27.kron.edgelist

  Triangle number = 106869298996

  Time = 308(s)

- twitter_rv.bin

  Triangle number = 34824916864

  Time = 967(s)

The time above is the best time in several times. It may be influenced by the memory cache or other unknown factors. It seems that the SSD's I/O speed is the main factors. It should be noted that the program use openmp to parallelize the caculation, so the time above is real world time instead of the CPU time.

## Algorithm analyzation
This project is finished with the help of the paper [High Performance Exact Triangle Counting on GPUs](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8000612).

We use Map-based triangle counting approach to caculate the triangles in the graph. We use both CPU and GPU to process the caculation. And we have many coding improvements to accelerate out caculation speed.

### CPU
In undirected graph, we can first build adjacency list, whose linked point ID is greater than current point ID. Then for each point with edge(s), we can first mark all it's neighbour(s) in a bitset. Then check all it's neighbours' neighbours, once you check a marked point, there is one more triangle in the graph.

The point ID in the graph files are not consecutive, so we have to hash them to build the adjacency list. And we must delete the multiple edges and self loops in the adjacency list. After that, we parallelized the counting procedure to accelerate the program.

### GPU
Just like CPU, we can parallelized the counting procedure. Because we have to deal with up to 128 million points, we can't use so much vectors to store the adjacency list(maybe I should rewrite the vector to adjust the requirement), than we have to use the edges list to store the information. After that, we must sort the edges list, and we can use GPU to accelerate the speed of sorting arrays. 

### Coding improvements
- We adjust the length of fread buffer, to get the balance between I/O speed and self loops processing speed.
- We use a 16GB RAM to hash all the unsigned int to a consecutive space, which represent point ID. In addition, we use openmp to parallelize the hashing.
- We use page-locked memory to store the edges list, otherwise the GPU's memory is not enough to sort the edges list.
- We change the ratio of points that GPU/CPU should process by the sum of degree. In fact, this approach get a good balance between GPU and CPU. The "ratio = 0.9" in the program is likely represent the ratio of GPU's compute capability to (CPU+GPU)'s total compute capability.

