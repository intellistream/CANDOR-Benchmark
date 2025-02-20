#!/bin/bash

export PATH=$PATH:/usr/local/cuda-10.0/bin

for i in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300
do
	./query_200_cos ../dataset/glove200/base.fvecs ../dataset/glove200/query.fvecs nsw ../dataset/glove200/base.fvecs_512_64_cos.nsw ../dataset/glove200/groundtruth.ivecs ${i} 10
done