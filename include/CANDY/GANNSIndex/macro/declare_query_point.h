#if GANNS
// 提前声明所有变量
float q1 = 0, q2 = 0, q3 = 0, q4 = 0, q5 = 0, q6 = 0, q7 = 0, q8 = 0;
float q9 = 0, q10 = 0, q11 = 0, q12 = 0, q13 = 0, q14 = 0, q15 = 0, q16 = 0;
float q17 = 0, q18 = 0, q19 = 0, q20 = 0, q21 = 0, q22 = 0, q23 = 0, q24 = 0;
float q25 = 0, q26 = 0, q27 = 0, q28 = 0, q29 = 0, q30 = 0;

if (CANDY::DIM > 0) {
    if (t_id < CANDY::DIM) {
        q1 = d_query[crt_point_id * CANDY::DIM + t_id];
    }
}
if (CANDY::DIM > 32) {
    if (t_id + 32 < CANDY::DIM) {
        q2 = d_query[crt_point_id * CANDY::DIM + t_id + 32];
    }
}
if (CANDY::DIM > 64) {
    if (t_id + 64 < CANDY::DIM) {
        q3 = d_query[crt_point_id * CANDY::DIM + t_id + 64];
    }
}
if (CANDY::DIM > 96) {
    if (t_id + 96 < CANDY::DIM) {
        q4 = d_query[crt_point_id * CANDY::DIM + t_id + 96];
    }
}
if (CANDY::DIM > 128) {
    if (t_id + 128 < CANDY::DIM) {
        q5 = d_query[crt_point_id * CANDY::DIM + t_id + 128];
    }
}
if (CANDY::DIM > 160) {
    if (t_id + 160 < CANDY::DIM) {
        q6 = d_query[crt_point_id * CANDY::DIM + t_id + 160];
    }
}
if (CANDY::DIM > 192) {
    if (t_id + 192 < CANDY::DIM) {
        q7 = d_query[crt_point_id * CANDY::DIM + t_id + 192];
    }
}
if (CANDY::DIM > 224) {
    if (t_id + 224 < CANDY::DIM) {
        q8 = d_query[crt_point_id * CANDY::DIM + t_id + 224];
    }
}
if (CANDY::DIM > 256) {
    if (t_id + 256 < CANDY::DIM) {
        q9 = d_query[crt_point_id * CANDY::DIM + t_id + 256];
    }
}
if (CANDY::DIM > 288) {
    if (t_id + 288 < CANDY::DIM) {
        q10 = d_query[crt_point_id * CANDY::DIM + t_id + 288];
    }
}
if (CANDY::DIM > 320) {
    if (t_id + 320 < CANDY::DIM) {
        q11 = d_query[crt_point_id * CANDY::DIM + t_id + 320];
    }
}
if (CANDY::DIM > 352) {
    if (t_id + 352 < CANDY::DIM) {
        q12 = d_query[crt_point_id * CANDY::DIM + t_id + 352];
    }
}
if (CANDY::DIM > 384) {
    if (t_id + 384 < CANDY::DIM) {
        q13 = d_query[crt_point_id * CANDY::DIM + t_id + 384];
    }
}
if (CANDY::DIM > 416) {
    if (t_id + 416 < CANDY::DIM) {
        q14 = d_query[crt_point_id * CANDY::DIM + t_id + 416];
    }
}
if (CANDY::DIM > 448) {
    if (t_id + 448 < CANDY::DIM) {
        q15 = d_query[crt_point_id * CANDY::DIM + t_id + 448];
    }
}
if (CANDY::DIM > 480) {
    if (t_id + 480 < CANDY::DIM) {
        q16 = d_query[crt_point_id * CANDY::DIM + t_id + 480];
    }
}
if (CANDY::DIM > 512) {
    if (t_id + 512 < CANDY::DIM) {
        q17 = d_query[crt_point_id * CANDY::DIM + t_id + 512];
    }
}
if (CANDY::DIM > 544) {
    if (t_id + 544 < CANDY::DIM) {
        q18 = d_query[crt_point_id * CANDY::DIM + t_id + 544];
    }
}
if (CANDY::DIM > 576) {
    if (t_id + 576 < CANDY::DIM) {
        q19 = d_query[crt_point_id * CANDY::DIM + t_id + 576];
    }
}
if (CANDY::DIM > 608) {
    if (t_id + 608 < CANDY::DIM) {
        q20 = d_query[crt_point_id * CANDY::DIM + t_id + 608];
    }
}
if (CANDY::DIM > 640) {
    if (t_id + 640 < CANDY::DIM) {
        q21 = d_query[crt_point_id * CANDY::DIM + t_id + 640];
    }
}
if (CANDY::DIM > 672) {
    if (t_id + 672 < CANDY::DIM) {
        q22 = d_query[crt_point_id * CANDY::DIM + t_id + 672];
    }
}
if (CANDY::DIM > 704) {
    if (t_id + 704 < CANDY::DIM) {
        q23 = d_query[crt_point_id * CANDY::DIM + t_id + 704];
    }
}
if (CANDY::DIM > 736) {
    if (t_id + 736 < CANDY::DIM) {
        q24 = d_query[crt_point_id * CANDY::DIM + t_id + 736];
    }
}
if (CANDY::DIM > 768) {
    if (t_id + 768 < CANDY::DIM) {
        q25 = d_query[crt_point_id * CANDY::DIM + t_id + 768];
    }
}
if (CANDY::DIM > 800) {
    if (t_id + 800 < CANDY::DIM) {
        q26 = d_query[crt_point_id * CANDY::DIM + t_id + 800];
    }
}
if (CANDY::DIM > 832) {
    if (t_id + 832 < CANDY::DIM) {
        q27 = d_query[crt_point_id * CANDY::DIM + t_id + 832];
    }
}
if (CANDY::DIM > 864) {
    if (t_id + 864 < CANDY::DIM) {
        q28 = d_query[crt_point_id * CANDY::DIM + t_id + 864];
    }
}
if (CANDY::DIM > 896) {
    if (t_id + 896 < CANDY::DIM) {
        q29 = d_query[crt_point_id * CANDY::DIM + t_id + 896];
    }
}
if (CANDY::DIM > 928) {
    if (t_id + 928 < CANDY::DIM) {
        q30 = d_query[crt_point_id * CANDY::DIM + t_id + 928];
    }
}
#endif