#if GANNS
// 提前声明所有 delta 和相关变量
float delta1 = 0, delta2 = 0, delta3 = 0, delta4 = 0, delta5 = 0, delta6 = 0, delta7 = 0, delta8 = 0;
float delta9 = 0, delta10 = 0, delta11 = 0, delta12 = 0, delta13 = 0, delta14 = 0, delta15 = 0, delta16 = 0;
float delta17 = 0, delta18 = 0, delta19 = 0, delta20 = 0, delta21 = 0, delta22 = 0, delta23 = 0, delta24 = 0;
float delta25 = 0, delta26 = 0, delta27 = 0, delta28 = 0, delta29 = 0, delta30 = 0;

//float p_l2_1 = 0, q_l2_1 = 0, p_l2_2 = 0, q_l2_2 = 0;  // Extend this for all `p_l2` and `q_l2` values if needed.
float p_l2_1 = 0, q_l2_1 = 0;
float p_l2_2 = 0, q_l2_2 = 0;
float p_l2_3 = 0, q_l2_3 = 0;
float p_l2_4 = 0, q_l2_4 = 0;
float p_l2_5 = 0, q_l2_5 = 0;
float p_l2_6 = 0, q_l2_6 = 0;
float p_l2_7 = 0, q_l2_7 = 0;
float p_l2_8 = 0, q_l2_8 = 0;
float p_l2_9 = 0, q_l2_9 = 0;
float p_l2_10 = 0, q_l2_10 = 0;
float p_l2_11 = 0, q_l2_11 = 0;
float p_l2_12 = 0, q_l2_12 = 0;
float p_l2_13 = 0, q_l2_13 = 0;
float p_l2_14 = 0, q_l2_14 = 0;
float p_l2_15 = 0, q_l2_15 = 0;
float p_l2_16 = 0, q_l2_16 = 0;
float p_l2_17 = 0, q_l2_17 = 0;
float p_l2_18 = 0, q_l2_18 = 0;
float p_l2_19 = 0, q_l2_19 = 0;
float p_l2_20 = 0, q_l2_20 = 0;
float p_l2_21 = 0, q_l2_21 = 0;
float p_l2_22 = 0, q_l2_22 = 0;
float p_l2_23 = 0, q_l2_23 = 0;
float p_l2_24 = 0, q_l2_24 = 0;
float p_l2_25 = 0, q_l2_25 = 0;
float p_l2_26 = 0, q_l2_26 = 0;
float p_l2_27 = 0, q_l2_27 = 0;
float p_l2_28 = 0, q_l2_28 = 0;
float p_l2_29 = 0, q_l2_29 = 0;
float p_l2_30 = 0, q_l2_30 = 0;

if (CANDY::USE_L2_DIST_) {
    if (CANDY::DIM > 0) delta1 = (p1 - q1) * (p1 - q1);
    if (CANDY::DIM > 32) delta2 = (p2 - q2) * (p2 - q2);
    if (CANDY::DIM > 64) delta3 = (p3 - q3) * (p3 - q3);
    if (CANDY::DIM > 96) delta4 = (p4 - q4) * (p4 - q4);
    if (CANDY::DIM > 128) delta5 = (p5 - q5) * (p5 - q5);
    if (CANDY::DIM > 160) delta6 = (p6 - q6) * (p6 - q6);
    if (CANDY::DIM > 192) delta7 = (p7 - q7) * (p7 - q7);
    if (CANDY::DIM > 224) delta8 = (p8 - q8) * (p8 - q8);
    if (CANDY::DIM > 256) delta9 = (p9 - q9) * (p9 - q9);
    if (CANDY::DIM > 288) delta10 = (p10 - q10) * (p10 - q10);
    if (CANDY::DIM > 320) delta11 = (p11 - q11) * (p11 - q11);
    if (CANDY::DIM > 352) delta12 = (p12 - q12) * (p12 - q12);
    if (CANDY::DIM > 384) delta13 = (p13 - q13) * (p13 - q13);
    if (CANDY::DIM > 416) delta14 = (p14 - q14) * (p14 - q14);
    if (CANDY::DIM > 448) delta15 = (p15 - q15) * (p15 - q15);
    if (CANDY::DIM > 480) delta16 = (p16 - q16) * (p16 - q16);
    if (CANDY::DIM > 512) delta17 = (p17 - q17) * (p17 - q17);
    if (CANDY::DIM > 544) delta18 = (p18 - q18) * (p18 - q18);
    if (CANDY::DIM > 576) delta19 = (p19 - q19) * (p19 - q19);
    if (CANDY::DIM > 608) delta20 = (p20 - q20) * (p20 - q20);
    if (CANDY::DIM > 640) delta21 = (p21 - q21) * (p21 - q21);
    if (CANDY::DIM > 672) delta22 = (p22 - q22) * (p22 - q22);
    if (CANDY::DIM > 704) delta23 = (p23 - q23) * (p23 - q23);
    if (CANDY::DIM > 736) delta24 = (p24 - q24) * (p24 - q24);
    if (CANDY::DIM > 768) delta25 = (p25 - q25) * (p25 - q25);
    if (CANDY::DIM > 800) delta26 = (p26 - q26) * (p26 - q26);
    if (CANDY::DIM > 832) delta27 = (p27 - q27) * (p27 - q27);
    if (CANDY::DIM > 864) delta28 = (p28 - q28) * (p28 - q28);
    if (CANDY::DIM > 896) delta29 = (p29 - q29) * (p29 - q29);
    if (CANDY::DIM > 928) delta30 = (p30 - q30) * (p30 - q30);
} else if (CANDY::USE_IP_DIST_) {
    if (CANDY::DIM > 0) delta1 = p1 * q1;
    if (CANDY::DIM > 32) delta2 = p2 * q2;
    if (CANDY::DIM > 64) delta3 = p3 * q3;
    if (CANDY::DIM > 96) delta4 = p4 * q4;
    if (CANDY::DIM > 128) delta5 = p5 * q5;
    if (CANDY::DIM > 160) delta6 = p6 * q6;
    if (CANDY::DIM > 192) delta7 = p7 * q7;
    if (CANDY::DIM > 224) delta8 = p8 * q8;
    if (CANDY::DIM > 256) delta9 = p9 * q9;
    if (CANDY::DIM > 288) delta10 = p10 * q10;
    if (CANDY::DIM > 320) delta11 = p11 * q11;
    if (CANDY::DIM > 352) delta12 = p12 * q12;
    if (CANDY::DIM > 384) delta13 = p13 * q13;
    if (CANDY::DIM > 416) delta14 = p14 * q14;
    if (CANDY::DIM > 448) delta15 = p15 * q15;
    if (CANDY::DIM > 480) delta16 = p16 * q16;
    if (CANDY::DIM > 512) delta17 = p17 * q17;
    if (CANDY::DIM > 544) delta18 = p18 * q18;
    if (CANDY::DIM > 576) delta19 = p19 * q19;
    if (CANDY::DIM > 608) delta20 = p20 * q20;
    if (CANDY::DIM > 640) delta21 = p21 * q21;
    if (CANDY::DIM > 672) delta22 = p22 * q22;
    if (CANDY::DIM > 704) delta23 = p23 * q23;
    if (CANDY::DIM > 736) delta24 = p24 * q24;
    if (CANDY::DIM > 768) delta25 = p25 * q25;
    if (CANDY::DIM > 800) delta26 = p26 * q26;
    if (CANDY::DIM > 832) delta27 = p27 * q27;
    if (CANDY::DIM > 864) delta28 = p28 * q28;
    if (CANDY::DIM > 896) delta29 = p29 * q29;
    if (CANDY::DIM > 928) delta30 = p30 * q30;
} else if (CANDY::USE_COS_DIST_) {
    if (CANDY::DIM > 0) {
        delta1 = p1 * q1;
        p_l2_1 = p1 * p1;
        q_l2_1 = q1 * q1;
    }
    if (CANDY::DIM > 32) {
        delta2 = p2 * q2;
        p_l2_2 = p2 * p2;
        q_l2_2 = q2 * q2;
    }
    if (CANDY::DIM > 64) {
        delta3 = p3 * q3;
        p_l2_3 = p3 * p3;
        q_l2_3 = q3 * q3;
    }
    if (CANDY::DIM > 96) {
        delta4 = p4 * q4;
        p_l2_4 = p4 * p4;
        q_l2_4 = q4 * q4;
    }
    if (CANDY::DIM > 128) {
        delta5 = p5 * q5;
        p_l2_5 = p5 * p5;
        q_l2_5 = q5 * q5;
    }
    if (CANDY::DIM > 160) {
        delta6 = p6 * q6;
        p_l2_6 = p6 * p6;
        q_l2_6 = q6 * q6;
    }
    if (CANDY::DIM > 192) {
        delta7 = p7 * q7;
        p_l2_7 = p7 * p7;
        q_l2_7 = q7 * q7;
    }
    if (CANDY::DIM > 224) {
        delta8 = p8 * q8;
        p_l2_8 = p8 * p8;
        q_l2_8 = q8 * q8;
    }
    if (CANDY::DIM > 256) {
        delta9 = p9 * q9;
        p_l2_9 = p9 * p9;
        q_l2_9 = q9 * q9;
    }
    if (CANDY::DIM > 288) {
        delta10 = p10 * q10;
        p_l2_10 = p10 * p10;
        q_l2_10 = q10 * q10;
    }
    if (CANDY::DIM > 320) {
        delta11 = p11 * q11;
        p_l2_11 = p11 * p11;
        q_l2_11 = q11 * q11;
    }
    if (CANDY::DIM > 352) {
        delta12 = p12 * q12;
        p_l2_12 = p12 * p12;
        q_l2_12 = q12 * q12;
    }
    if (CANDY::DIM > 384) {
        delta13 = p13 * q13;
        p_l2_13 = p13 * p13;
        q_l2_13 = q13 * q13;
    }
    if (CANDY::DIM > 416) {
        delta14 = p14 * q14;
        p_l2_14 = p14 * p14;
        q_l2_14 = q14 * q14;
    }
    if (CANDY::DIM > 448) {
        delta15 = p15 * q15;
        p_l2_15 = p15 * p15;
        q_l2_15 = q15 * q15;
    }
    if (CANDY::DIM > 480) {
        delta16 = p16 * q16;
        p_l2_16 = p16 * p16;
        q_l2_16 = q16 * q16;
    }
    if (CANDY::DIM > 512) {
        delta17 = p17 * q17;
        p_l2_17 = p17 * p17;
        q_l2_17 = q17 * q17;
    }
    if (CANDY::DIM > 544) {
        delta18 = p18 * q18;
        p_l2_18 = p18 * p18;
        q_l2_18 = q18 * q18;
    }
    if (CANDY::DIM > 576) {
        delta19 = p19 * q19;
        p_l2_19 = p19 * p19;
        q_l2_19 = q19 * q19;
    }
    if (CANDY::DIM > 608) {
        delta20 = p20 * q20;
        p_l2_20 = p20 * p20;
        q_l2_20 = q20 * q20;
    }
    if (CANDY::DIM > 640) {
        delta21 = p21 * q21;
        p_l2_21 = p21 * p21;
        q_l2_21 = q21 * q21;
    }
    if (CANDY::DIM > 672) {
        delta22 = p22 * q22;
        p_l2_22 = p22 * p22;
        q_l2_22 = q22 * q22;
    }
    if (CANDY::DIM > 704) {
        delta23 = p23 * q23;
        p_l2_23 = p23 * p23;
        q_l2_23 = q23 * q23;
    }
    if (CANDY::DIM > 736) {
        delta24 = p24 * q24;
        p_l2_24 = p24 * p24;
        q_l2_24 = q24 * q24;
    }
    if (CANDY::DIM > 768) {
        delta25 = p25 * q25;
        p_l2_25 = p25 * p25;
        q_l2_25 = q25 * q25;
    }
    if (CANDY::DIM > 800) {
        delta26 = p26 * q26;
        p_l2_26 = p26 * p26;
        q_l2_26 = q26 * q26;
    }
    if (CANDY::DIM > 832) {
        delta27 = p27 * q27;
        p_l2_27 = p27 * p27;
        q_l2_27 = q27 * q27;
    }
    if (CANDY::DIM > 864) {
        delta28 = p28 * q28;
        p_l2_28 = p28 * p28;
        q_l2_28 = q28 * q28;
    }
    if (CANDY::DIM > 896) {
        delta29 = p29 * q29;
        p_l2_29 = p29 * p29;
        q_l2_29 = q29 * q29;
    }
    if (CANDY::DIM > 928) {
        delta30 = p30 * q30;
        p_l2_30 = p30 * p30;
        q_l2_30 = q30 * q30;
    }

    // Extend similar logic for other deltas, p_l2, and q_l2 values
}
#endif