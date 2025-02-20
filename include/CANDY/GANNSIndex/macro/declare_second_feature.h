#if GANNS
// 提前声明所有可能用到的变量
float p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0;
float p9 = 0, p10 = 0, p11 = 0, p12 = 0, p13 = 0, p14 = 0, p15 = 0, p16 = 0;
float p17 = 0, p18 = 0, p19 = 0, p20 = 0, p21 = 0, p22 = 0, p23 = 0, p24 = 0;
float p25 = 0, p26 = 0, p27 = 0, p28 = 0, p29 = 0, p30 = 0;

if (CANDY::DIM > 0) {
    if (t_id < CANDY::DIM) {
        p1 = d_data[target_point_id * CANDY::DIM + t_id];
    }
}
if (CANDY::DIM > 32) {
    if (t_id + 32 < CANDY::DIM) {
        p2 = d_data[target_point_id * CANDY::DIM + t_id + 32];
    }
}
if (CANDY::DIM > 64) {
    if (t_id + 64 < CANDY::DIM) {
        p3 = d_data[target_point_id * CANDY::DIM + t_id + 64];
    }
}
if (CANDY::DIM > 96) {
    if (t_id + 96 < CANDY::DIM) {
        p4 = d_data[target_point_id * CANDY::DIM + t_id + 96];
    }
}
if (CANDY::DIM > 128) {
    if (t_id + 128 < CANDY::DIM) {
        p5 = d_data[target_point_id * CANDY::DIM + t_id + 128];
    }
}
if (CANDY::DIM > 160) {
    if (t_id + 160 < CANDY::DIM) {
        p6 = d_data[target_point_id * CANDY::DIM + t_id + 160];
    }
}
if (CANDY::DIM > 192) {
    if (t_id + 192 < CANDY::DIM) {
        p7 = d_data[target_point_id * CANDY::DIM + t_id + 192];
    }
}
if (CANDY::DIM > 224) {
    if (t_id + 224 < CANDY::DIM) {
        p8 = d_data[target_point_id * CANDY::DIM + t_id + 224];
    }
}
if (CANDY::DIM > 256) {
    if (t_id + 256 < CANDY::DIM) {
        p9 = d_data[target_point_id * CANDY::DIM + t_id + 256];
    }
}
if (CANDY::DIM > 288) {
    if (t_id + 288 < CANDY::DIM) {
        p10 = d_data[target_point_id * CANDY::DIM + t_id + 288];
    }
}
if (CANDY::DIM > 320) {
    if (t_id + 320 < CANDY::DIM) {
        p11 = d_data[target_point_id * CANDY::DIM + t_id + 320];
    }
}
if (CANDY::DIM > 352) {
    if (t_id + 352 < CANDY::DIM) {
        p12 = d_data[target_point_id * CANDY::DIM + t_id + 352];
    }
}
if (CANDY::DIM > 384) {
    if (t_id + 384 < CANDY::DIM) {
        p13 = d_data[target_point_id * CANDY::DIM + t_id + 384];
    }
}
if (CANDY::DIM > 416) {
    if (t_id + 416 < CANDY::DIM) {
        p14 = d_data[target_point_id * CANDY::DIM + t_id + 416];
    }
}
if (CANDY::DIM > 448) {
    if (t_id + 448 < CANDY::DIM) {
        p15 = d_data[target_point_id * CANDY::DIM + t_id + 448];
    }
}
if (CANDY::DIM > 480) {
    if (t_id + 480 < CANDY::DIM) {
        p16 = d_data[target_point_id * CANDY::DIM + t_id + 480];
    }
}
if (CANDY::DIM > 512) {
    if (t_id + 512 < CANDY::DIM) {
        p17 = d_data[target_point_id * CANDY::DIM + t_id + 512];
    }
}
if (CANDY::DIM > 544) {
    if (t_id + 544 < CANDY::DIM) {
        p18 = d_data[target_point_id * CANDY::DIM + t_id + 544];
    }
}
if (CANDY::DIM > 576) {
    if (t_id + 576 < CANDY::DIM) {
        p19 = d_data[target_point_id * CANDY::DIM + t_id + 576];
    }
}
if (CANDY::DIM > 608) {
    if (t_id + 608 < CANDY::DIM) {
        p20 = d_data[target_point_id * CANDY::DIM + t_id + 608];
    }
}
if (CANDY::DIM > 640) {
    if (t_id + 640 < CANDY::DIM) {
        p21 = d_data[target_point_id * CANDY::DIM + t_id + 640];
    }
}
if (CANDY::DIM > 672) {
    if (t_id + 672 < CANDY::DIM) {
        p22 = d_data[target_point_id * CANDY::DIM + t_id + 672];
    }
}
if (CANDY::DIM > 704) {
    if (t_id + 704 < CANDY::DIM) {
        p23 = d_data[target_point_id * CANDY::DIM + t_id + 704];
    }
}
if (CANDY::DIM > 736) {
    if (t_id + 736 < CANDY::DIM) {
        p24 = d_data[target_point_id * CANDY::DIM + t_id + 736];
    }
}
if (CANDY::DIM > 768) {
    if (t_id + 768 < CANDY::DIM) {
        p25 = d_data[target_point_id * CANDY::DIM + t_id + 768];
    }
}
if (CANDY::DIM > 800) {
    if (t_id + 800 < CANDY::DIM) {
        p26 = d_data[target_point_id * CANDY::DIM + t_id + 800];
    }
}
if (CANDY::DIM > 832) {
    if (t_id + 832 < CANDY::DIM) {
        p27 = d_data[target_point_id * CANDY::DIM + t_id + 832];
    }
}
if (CANDY::DIM > 864) {
    if (t_id + 864 < CANDY::DIM) {
        p28 = d_data[target_point_id * CANDY::DIM + t_id + 864];
    }
}
if (CANDY::DIM > 896) {
    if (t_id + 896 < CANDY::DIM) {
        p29 = d_data[target_point_id * CANDY::DIM + t_id + 896];
    }
}
if (CANDY::DIM > 928) {
    if (t_id + 928 < CANDY::DIM) {
        p30 = d_data[target_point_id * CANDY::DIM + t_id + 928];
    }
}
#endif