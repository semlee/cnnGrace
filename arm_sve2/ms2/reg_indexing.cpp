    ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
    ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
    ifhp = ifh + 2 * pad_h_in;
    ifwp = ifw + 2 * pad_w_in;
    ofhp = ofh + 2 * pad_h_out;
    ofwp = ofw + 2 * pad_w_out;
    

    size_t inputIndex =     img * nIfm * ifhp * ifwp + 
                            ifm * ifhp * ifwp + 
                            (ij + kj) * ifwp + 
                            (ii + ki);
    size_t outputIndex =    img * nOfm * ofhp * ofwp + 
                            ofm * ofhp * ofwp + 
                            oj * ofwp + 
                            oi;
    size_t filterIndex =    ofm * nIfm * kh * kw + 
                            ifm * kh * kw + 
                            kj * kw + 
                            ki;
    
    size_t inputIndex =     n * C * ifhp * ifwp + 
                            c * ifhp * ifwp + 
                            (ijo + r) * ifwp + 
                            (iio + s);
    size_t outputIndex =    n * K * ofhp * ofwp + 
                            k * ofhp * ofwp + 
                            oj * ofwp + 
                            oi;
    size_t filterIndex =    k * C * R * S + 
                            c * R * S + 
                            r * S + 
                            s;

    size_t inputIndex =     n * C * ifhp * ifwp + 
                            (c_b * VLEN + c) * ifhp * ifwp + 
                            (ijo + r) * ifwp + 
                            (iio + s);
    size_t outputIndex =    n * K * ofhp * ofwp + 
                            (k_b * VLEN + k) * ofhp * ofwp + 
                            oj * ofwp + 
                            oi;
    size_t filterIndex =    (k_b * VLEN + k) * C * kh * kw + 
                            (c_b * VLEN + c) * kh * kw + 
                            r * kw + 
                            s;

    for (img = 0; img < nImg; ++img) {
        for (ofm = 0; ofm < nOfm; ++ofm) {
            for (ifm = 0; ifm < nIfm; ++ifm) {
                for (oj = 0; oj < ofh; ++oj) {
                    ij = oj * stride_h - pad_h;
                    for (oi = 0; oi < ofw; ++oi) {
                        ii = oi * stride_w - pad_w;
                        for (kj = 0; kj < kh; ++kj) {
                            if (ij+kj < 0 || ij+kj >= ifh) continue;
                            for (ki = 0; ki < kw; ++ki) {
                                if (ii+ki < 0 || ii+ki >= ifw) continue;


    for (n = 0; n < N; ++n) {
        for (k = 0; k < K; ++k) {
            for (c = 0; c < C; ++c) {
                for (oj = 0; oj < P; ++oj) {
                    ij = oj * stride_h - pad_h;
                    for (oi = 0; oi < Q; ++oi) {
                        ii = oi * stride_w - pad_w;
                        for (r = 0; r < R; ++r) {
                            if (ij+r < 0 || ij+r >= ifh) continue;
                            for (s = 0; s < S; ++s) {
                                if (ii+s < 0 || ii+s >= ifw) continue;
    
    for (img = 0; img < nImg; img++) {
        for (ofm_b = 0; ofm_b < nOfm_b; ofm_b++) {
            for (ifm_b = 0; ifm_b < nIfm_b; ifm_b++) {
                for (oj_b = 0; oj_b < ofh_b; oj_b++) {
                    oj = oj_b * RB_p;
                    ij = oj * stride_h;
                    for (oi_b = 0; oi_b < ofw_b; oi_b++) {
                        oi = oi_b * RB_p;
                        ii = oi * stride_h;
                        for (kj = 0; kj < kh; kj++) {
                            if (ij + kj < 0 || ij + kj >= ifh) continue;
                            for (ki = 0; ki < kw; ki++) {
                                if (ii + ki < 0 || ii + ki >= ifw) continue;
                                for (ofm = 0; ofm <= VLEN; ofm++) {
                                    for (ifm = 0; ifm <= VLEN; ifm++) {
                                        for (p = 0; p <= RB_p; p++) {
                                            for (q = 0; q <= RB_q; q++) {
                                                ijo = ij + stride_h * p - pad_h;
                                                iio = ii + stride_w * q - pad_q;

    

    for (n = 0; n < N; n++) {
        for (k_b = 0; k_b < K_b; k_b++) {
            for (c_b = 0; c_b < C_b; c_b++) {
                for (oj_b = 0; oj < P_b; oj++) {
                    oj = oj_b * RB_p;
                    ij = oj * stride_h;
                    for (oi_b = 0; oi < Q_b; oi++) {
                        oi = oi_b * RB_p;
                        ii = oi * stride_h;
                        for (r = 0; r < R; r++) {
                            if (ij + r < 0 || ij + r >= ifh) continue;
                            for (s = 0; s < S; s++) {
                                if (ii + s < 0 || ii + s >= ifw) continue;
                                for (c = 0; c <= VLEN; c++) {
                                    for (k = 0; k <= VLEN; k++) {
                                        for (p = 0; p <= RB_p; p++) {
                                            for (q = 0; q <= RB_q; q++) {
                                                ijo = ij + stride_h * p - pad_h;
                                                iio = ii + stride_w * q - pad_q;

/////////////////////////////////////////////////////////////////////////////////////
//type 1
    int nIfm_b = nIfm/VLEN;
    int nOfm = nOfm/VLEN;
    int ofh_b = ofh/RB_p;
    int ofw_b = ofw/RB_q;
    int img, ofm_b, ifm_b, oj_b, oj, ij, oi_b, oi, ii, kj, ki, ofm, ifm, p, q, ijo, iio;

    for (img = 0; img < nImg; img++) {
        for (ofm_b = 0; ofm_b < nOfm_b; ofm_b++) {
            for (ifm_b = 0; ifm_b < nIfm_b; ifm_b++) {
                for (oj_b = 0; oj_b < ofh_b; oj_b++) {
                    oj = oj_b * RB_p;
                    ij = oj * stride_h;
                    for (oi_b = 0; oi_b < ofw_b; oi_b++) {
                        oi = oi_b * RB_p;
                        ii = oi * stride_h;
                        for (kj = 0; kj < kh; kj++) {
                            if (ij + kj < 0 || ij + kj >= ifh) continue;
                            for (ki = 0; ki < kw; ki++) {
                                if (ii + ki < 0 || ii + ki >= ifw) continue;
                                for (ofm = 0; ofm <= VLEN; ofm++) {
                                    for (ifm = 0; ifm <= VLEN; ifm++) {
                                        for (p = 0; p <= RB_p; p++) {
                                            for (q = 0; q <= RB_q; q++) {
                                                ijo = ij + stride_h * p - pad_h;
                                                iio = ii + stride_w * q - pad_w;
                                                size_t inputIndex =     img * nIfm * ifhp * ifwp + 
                                                                        (ifm_b * VLEN + ifm) * ifhp * ifwp + 
                                                                        (ij + kj) * ifwp + 
                                                                        (ii + ki);
                                                size_t outputIndex =    img * nOfm * ofhp * ofwp + 
                                                                        (ofm_b * VLEN + ofm) * ofhp * ofwp + 
                                                                        oj * ofwp + 
                                                                        oi;
                                                size_t filterIndex =    (ofm_b * VLEN + ofm) * nIfm * kh * kw + 
                                                                        (ifm_b * VLEN + ifm) * kh * kw + 
                                                                        kj * kw + 
                                                                        ki;

/////////////////////////////////////////////////////////////////////////////////////
//type 2
    int N         = param->nImg;
    int C         = param->nIfm;
    int K         = param->nOfm;
    int ifhp      = param->ifhp;
    int ifwp      = param->ifwp;
    int ofhp      = param->ofhp;
    int ofwp      = param->ofwp;
    int ifh       = param->ifh;
    int ifw       = param->ifw;
    int P         = param->ofh;
    int Q         = param->ofw;
    int pad_h     = param->pad_h;
    int pad_w     = param->pad_w;
    int pad_h_in  = param->pad_h_in;
    int pad_w_in  = param->pad_w_in;
    int pad_h_out = param->pad_h_out;
    int pad_w_out = param->pad_w_out;
    int R         = param->kh;
    int S         = param->kw;
    int stride_h  = param->stride_h;
    int stride_w  = param->stride_w;
    int VLEN      = param->VLEN;
    int RB_p      = param->RB_p;
    int RB_q      = param->RB_q;

    int C_b = C/VLEN;
    int K_b = K/VLEN;
    int P_b = P/RB_p;
    int Q_b = Q/RB_q;
    int n, k_b, c_b, oj_b, oj, ij, oi_b, oi, ii, r, s, c, k, p, q, ijo, iio;

    for (n = 0; n < N; n++) {
        for (k_b = 0; k_b < K_b; k_b++) {
            for (c_b = 0; c_b < C_b; c_b++) {
                for (oj_b = 0; oj < P_b; oj++) {
                    oj = oj_b * RB_p;
                    ij = oj * stride_h;
                    for (oi_b = 0; oi < Q_b; oi++) {
                        oi = oi_b * RB_p;
                        ii = oi * stride_h;
                        for (r = 0; r < R; r++) {
                            if (ij + r < 0 || ij + r >= ifh) continue;
                            for (s = 0; s < S; s++) {
                                if (ii + s < 0 || ii + s >= ifw) continue;
                                for (c = 0; c <= VLEN; c++) {
                                    for (k = 0; k <= VLEN; k++) {
                                        for (p = 0; p <= RB_p; p++) {
                                            for (q = 0; q <= RB_q; q++) {
                                                ijo = ij + stride_h * p - pad_h;
                                                iio = ii + stride_w * q - pad_w;
                                                size_t inputIndex =     n * C * ifhp * ifwp + 
                                                                        (c_b * VLEN + c) * ifhp * ifwp + 
                                                                        (ijo + r) * ifwp + 
                                                                        (iio + s);
                                                size_t outputIndex =    n * K * ofhp * ofwp + 
                                                                        (k_b * VLEN + k) * ofhp * ofwp + 
                                                                        oj * ofwp + 
                                                                        oi;
                                                size_t filterIndex =    (k_b * VLEN + k) * C * R * S + 
                                                                        (c_b * VLEN + c) * R * S + 
                                                                        r * S + 
                                                                        s;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

#if defined(USE_FUSED_RELU) || defined(USE_FUSED_BIAS_RELU)
            // Apply ReLU activation function
            for (int oj = 0; oj < P; oj++) {
                for (int oi = 0; oi < Q; oi++) {
                    int reluIndex = n * K * ofhp * ofwp +
                                    k_b * K_b * ofhp * ofwp +
                                    oj * ofwp +
                                    oi;
                    output[reluIndex] = (output[reluIndex] < 0.0f) ? 0.0f : output[reluIndex];
                }
            }
#endif
        }
    }
        