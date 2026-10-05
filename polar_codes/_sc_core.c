#include <math.h>
#include <stdlib.h>
#include <string.h>

static double f_box(double a, double b) {
    double sa = (a >= 0) ? 1.0 : -1.0;
    double sb = (b >= 0) ? 1.0 : -1.0;
    double ma = fabs(a);
    double mb = fabs(b);
    return sa * sb * (ma < mb ? ma : mb);
}

static double g_box(double a, double b, int u) {
    return ((1 - 2 * u) * a) + b;
}

static void decode_node(const double *llr, int len, const int *frozen, int *uhat, int offset) {
    if (len == 1) {
        int idx = offset;
        if (frozen[idx]) {
            uhat[idx] = 0;
        } else {
            uhat[idx] = (llr[0] >= 0.0) ? 0 : 1;
        }
        return;
    }
    int half = len / 2;
    double *llr_left = (double *)malloc(half * sizeof(double));
    double *llr_right = (double *)malloc(half * sizeof(double));
    for (int i = 0; i < half; i++) {
        llr_left[i] = f_box(llr[i], llr[i + half]);
    }
    decode_node(llr_left, half, frozen, uhat, offset);
    for (int i = 0; i < half; i++) {
        llr_right[i] = g_box(llr[i], llr[i + half], uhat[offset + i]);
    }
    decode_node(llr_right, half, frozen, uhat, offset + half);
    free(llr_left);
    free(llr_right);
}

void polar_sc_decode(const double *llr, int N, const int *frozen, int *uhat) {
    decode_node(llr, N, frozen, uhat, 0);
}
