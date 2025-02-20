#!/usr/bin/env bash

dim=$1
dis=$2

DIR="build_instance"

if [ -d "$DIR" ]; then
    rm -rf ${DIR}
fi

mkdir build_instance || true

cp template/*.h build_instance
cp template/*.cu build_instance
cp -R template/macro build_instance
cp -R template/graph_index build_instance
cp template/Makefile build_instance

cd build_instance

sed -i "s/PLACE_HOLDER_DIM/${dim}/g" kernel_local_graph_construction.h

if [ "${dis}" = "l2" ]; then
	make build
elif [ "${dis}" = "cos" ]; then
	make build DISTTYPE=USE_COS_DIST_
elif [ "${dis}" = "ip" ]; then
	make build DISTTYPE=USE_IP_DIST_
fi

instance_name="build_${dim}_${dis}"

mv build ${instance_name}
cp ${instance_name} ..
rm ${instance_name}