
float dist = 0;  // 提前声明 dist
float p_l2 = 0;  // 如果 USE_COS_DIST_ 会用到
float q_l2 = 0;  // 如果 USE_COS_DIST_ 会用到
#if GANNS
if (CANDY::USE_L2_DIST_) {
    if (CANDY::DIM > 0) dist += delta1;
    if (CANDY::DIM > 32) dist += delta2;
    if (CANDY::DIM > 64) dist += delta3;
    if (CANDY::DIM > 96) dist += delta4;
    if (CANDY::DIM > 128) dist += delta5;
    if (CANDY::DIM > 160) dist += delta6;
    if (CANDY::DIM > 192) dist += delta7;
    if (CANDY::DIM > 224) dist += delta8;
    if (CANDY::DIM > 256) dist += delta9;
    if (CANDY::DIM > 288) dist += delta10;
    if (CANDY::DIM > 320) dist += delta11;
    if (CANDY::DIM > 352) dist += delta12;
    if (CANDY::DIM > 384) dist += delta13;
    if (CANDY::DIM > 416) dist += delta14;
    if (CANDY::DIM > 448) dist += delta15;
    if (CANDY::DIM > 480) dist += delta16;
    if (CANDY::DIM > 512) dist += delta17;
    if (CANDY::DIM > 544) dist += delta18;
    if (CANDY::DIM > 576) dist += delta19;
    if (CANDY::DIM > 608) dist += delta20;
    if (CANDY::DIM > 640) dist += delta21;
    if (CANDY::DIM > 672) dist += delta22;
    if (CANDY::DIM > 704) dist += delta23;
    if (CANDY::DIM > 736) dist += delta24;
    if (CANDY::DIM > 768) dist += delta25;
    if (CANDY::DIM > 800) dist += delta26;
    if (CANDY::DIM > 832) dist += delta27;
    if (CANDY::DIM > 864) dist += delta28;
    if (CANDY::DIM > 896) dist += delta29;
    if (CANDY::DIM > 928) dist += delta30;
} else if (CANDY::USE_IP_DIST_) {
    if (CANDY::DIM > 0) dist += delta1;
    if (CANDY::DIM > 32) dist += delta2;
    if (CANDY::DIM > 64) dist += delta3;
    if (CANDY::DIM > 96) dist += delta4;
    if (CANDY::DIM > 128) dist += delta5;
    if (CANDY::DIM > 160) dist += delta6;
    if (CANDY::DIM > 192) dist += delta7;
    if (CANDY::DIM > 224) dist += delta8;
    if (CANDY::DIM > 256) dist += delta9;
    if (CANDY::DIM > 288) dist += delta10;
    if (CANDY::DIM > 320) dist += delta11;
    if (CANDY::DIM > 352) dist += delta12;
    if (CANDY::DIM > 384) dist += delta13;
    if (CANDY::DIM > 416) dist += delta14;
    if (CANDY::DIM > 448) dist += delta15;
    if (CANDY::DIM > 480) dist += delta16;
    if (CANDY::DIM > 512) dist += delta17;
    if (CANDY::DIM > 544) dist += delta18;
    if (CANDY::DIM > 576) dist += delta19;
    if (CANDY::DIM > 608) dist += delta20;
    if (CANDY::DIM > 640) dist += delta21;
    if (CANDY::DIM > 672) dist += delta22;
    if (CANDY::DIM > 704) dist += delta23;
    if (CANDY::DIM > 736) dist += delta24;
    if (CANDY::DIM > 768) dist += delta25;
    if (CANDY::DIM > 800) dist += delta26;
    if (CANDY::DIM > 832) dist += delta27;
    if (CANDY::DIM > 864) dist += delta28;
    if (CANDY::DIM > 896) dist += delta29;
    if (CANDY::DIM > 928) dist += delta30;
} else if (CANDY::USE_COS_DIST_) {
    if (CANDY::DIM > 0) {
        dist += delta1;
        p_l2 += p_l2_1;
        q_l2 += q_l2_1;
    }
    if (CANDY::DIM > 32) {
        dist += delta2;
        p_l2 += p_l2_2;
        q_l2 += q_l2_2;
    }
    if (CANDY::DIM > 64) {
        dist += delta3;
        p_l2 += p_l2_3;
        q_l2 += q_l2_3;
    }
    if (CANDY::DIM > 96) {
        dist += delta4;
        p_l2 += p_l2_4;
        q_l2 += q_l2_4;
    }
    if (CANDY::DIM > 128) {
        dist += delta5;
        p_l2 += p_l2_5;
        q_l2 += q_l2_5;
    }
    if (CANDY::DIM > 160) {
        dist += delta6;
        p_l2 += p_l2_6;
        q_l2 += q_l2_6;
    }
    if (CANDY::DIM > 192) {
        dist += delta7;
        p_l2 += p_l2_7;
        q_l2 += q_l2_7;
    }
    if (CANDY::DIM > 224) {
        dist += delta8;
        p_l2 += p_l2_8;
        q_l2 += q_l2_8;
    }
    if (CANDY::DIM > 256) {
        dist += delta9;
        p_l2 += p_l2_9;
        q_l2 += q_l2_9;
    }
    if (CANDY::DIM > 288) {
        dist += delta10;
        p_l2 += p_l2_10;
        q_l2 += q_l2_10;
    }
    if (CANDY::DIM > 320) {
        dist += delta11;
        p_l2 += p_l2_11;
        q_l2 += q_l2_11;
    }
    if (CANDY::DIM > 352) {
        dist += delta12;
        p_l2 += p_l2_12;
        q_l2 += q_l2_12;
    }
    if (CANDY::DIM > 384) {
        dist += delta13;
        p_l2 += p_l2_13;
        q_l2 += q_l2_13;
    }
    if (CANDY::DIM > 416) {
        dist += delta14;
        p_l2 += p_l2_14;
        q_l2 += q_l2_14;
    }
    if (CANDY::DIM > 448) {
        dist += delta15;
        p_l2 += p_l2_15;
        q_l2 += q_l2_15;
    }
    if (CANDY::DIM > 480) {
        dist += delta16;
        p_l2 += p_l2_16;
        q_l2 += q_l2_16;
    }
    if (CANDY::DIM > 512) {
        dist += delta17;
        p_l2 += p_l2_17;
        q_l2 += q_l2_17;
    }
    if (CANDY::DIM > 544) {
        dist += delta18;
        p_l2 += p_l2_18;
        q_l2 += q_l2_18;
    }
    if (CANDY::DIM > 576) {
        dist += delta19;
        p_l2 += p_l2_19;
        q_l2 += q_l2_19;
    }
    if (CANDY::DIM > 608) {
        dist += delta20;
        p_l2 += p_l2_20;
        q_l2 += q_l2_20;
    }
    if (CANDY::DIM > 640) {
        dist += delta21;
        p_l2 += p_l2_21;
        q_l2 += q_l2_21;
    }
    if (CANDY::DIM > 672) {
        dist += delta22;
        p_l2 += p_l2_22;
        q_l2 += q_l2_22;
    }
    if (CANDY::DIM > 704) {
        dist += delta23;
        p_l2 += p_l2_23;
        q_l2 += q_l2_23;
    }
    if (CANDY::DIM > 736) {
        dist += delta24;
        p_l2 += p_l2_24;
        q_l2 += q_l2_24;
    }
    if (CANDY::DIM > 768) {
        dist += delta25;
        p_l2 += p_l2_25;
        q_l2 += q_l2_25;
    }
    if (CANDY::DIM > 800) {
        dist += delta26;
        p_l2 += p_l2_26;
        q_l2 += q_l2_26;
    }
    if (CANDY::DIM > 832) {
        dist += delta27;
        p_l2 += p_l2_27;
        q_l2 += q_l2_27;
    }
    if (CANDY::DIM > 864) {
        dist += delta28;
        p_l2 += p_l2_28;
        q_l2 += q_l2_28;
    }
    if (CANDY::DIM > 896) {
        dist += delta29;
        p_l2 += p_l2_29;
        q_l2 += q_l2_29;
    }
    if (CANDY::DIM > 928) {
        dist += delta30;
        p_l2 += p_l2_30;
        q_l2 += q_l2_30;
    }
}
#endif