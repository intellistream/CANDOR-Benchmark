#!/usr/bin/env bash

dim=$1
dis=$2

DIR="query_instance"

if [ -d "$DIR" ]; then
    rm -rf ${DIR}
fi

mkdir query_instance || true

cp template/*.h query_instance
cp template/*.cu query_instance
cp -R template/macro query_instance
cp -R template/graph_index query_instance
cp template/Makefile query_instance

cd query_instance

sed -i "s/PLACE_HOLDER_DIM/${dim}/g" kernel_local_graph_construction.h

if [ "${dis}" = "l2" ]; then
	make query mode=${mode}
elif [ "${dis}" = "cos" ]; then
	make query DISTTYPE=USE_COS_DIST_
elif [ "${dis}" = "ip" ]; then
	make query DISTTYPE=USE_IP_DIST_
fi

instance_name="query_${dim}_${dis}"

mv query ${instance_name}
cp ${instance_name} ..
rm ${instance_name}