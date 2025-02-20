#!/usr/bin/env bash

DIR="template"

if [ -d "$DIR" ]; then
    rm -rf ${DIR}
fi

mkdir template || true

cp *.h template
cp *.cu template
cp -R macro template
cp -R graph_index template
cp Makefile template

cd template

nl='\n'

while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}DECLARE_QUERY_POINT_"
    sed -i "s@DECLARE_QUERY_POINT_@${line_with_tail_symbol}@" *.h
done < macro/declare_query_point.h
sed -i "s@DECLARE_QUERY_POINT_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}DECLARE_FEATURE_"
    sed -i "s@DECLARE_FEATURE_@${line_with_tail_symbol}@" *.h
done < macro/declare_feature.h
sed -i "s@DECLARE_FEATURE_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}DECLARE_SECOND_FEATURE_"
    sed -i "s@DECLARE_SECOND_FEATURE_@${line_with_tail_symbol}@" *.h
done < macro/declare_second_feature.h
sed -i "s@DECLARE_SECOND_FEATURE_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}COMPUTATION_"
    sed -i "s@COMPUTATION_@${line_with_tail_symbol}@" *.h
done < macro/computation.h
sed -i "s@COMPUTATION_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}SUM_UP_"
    sed -i "s@SUM_UP_@${line_with_tail_symbol}@" *.h
done < macro/sum_up.h
sed -i "s@SUM_UP_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}WITHOUT_FLAG_BITONIC_MERGE_"
    sed -i "s@WITHOUT_FLAG_BITONIC_MERGE_@${line_with_tail_symbol}@" *.h
done < macro/without_flag_bitonic_merge.h
sed -i "s@WITHOUT_FLAG_BITONIC_MERGE_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}WITHIN_WARP_"
    sed -i "s@WITHIN_WARP_@${line_with_tail_symbol}@" *.h
done < macro/within_warp.h
sed -i "s@WITHIN_WARP_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}BINARY_SEARCH_"
    sed -i "s@BINARY_SEARCH_@${line_with_tail_symbol}@" *.h
done < macro/binary_search.h
sed -i "s@BINARY_SEARCH_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}BITONIC_MERGE_"
    sed -i "s@BITONIC_MERGE_@${line_with_tail_symbol}@" *.h
done < macro/bitonic_merge.h
sed -i "s@BITONIC_MERGE_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}BITONIC_SORT_ON_CRT_BESTS_"
    sed -i "s@BITONIC_SORT_ON_CRT_BESTS_@${line_with_tail_symbol}@" *.h
done < macro/bitonic_sort_on_crt_bests.h
sed -i "s@BITONIC_SORT_ON_CRT_BESTS_@@" *.h


while IFS= read -r line || [[ -n "$line" ]]; do
	line_with_tail_symbol="${line}${nl}BITONIC_SORT_ON_NEIGHBORS_"
    sed -i "s@BITONIC_SORT_ON_NEIGHBORS_@${line_with_tail_symbol}@" *.h
done < macro/bitonic_sort_on_neighbors.h
sed -i "s@BITONIC_SORT_ON_NEIGHBORS_@@" *.h