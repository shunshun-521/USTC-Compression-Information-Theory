#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int sgn(double v) { return (v > 0) - (v < 0); }

static void f_vec(const double* x, const double* y, double* z, uint16_t len) {
  for (uint16_t i = 0; i < len; i++) {
    double L0 = x[i];
    double L1 = y[i];
    double a0 = fabs(L0);
    double a1 = fabs(L1);
    double s = (double)sgn(L0) * (double)sgn(L1);
    z[i] = s * (a0 < a1 ? a0 : a1);
  }
}

static void g_vec(const uint8_t* b, const double* x, const double* y, double* z, uint16_t len) {
  for (uint16_t i = 0; i < len; i++) {
    double V = -2.0 * b[i] + 1.0;
    z[i] = y[i] + V * x[i];
  }
}

static void xor_vec(const uint8_t* x, const uint8_t* y, uint8_t* z, uint32_t len) {
  for (uint32_t i = 0; i < len; i++) {
    z[i] = x[i] ^ y[i];
  }
}

typedef struct {
  uint8_t  n;
  uint16_t N;
  double** llr0;
  double** llr1;
  uint8_t* est_bit;
  uint8_t* frozen;
  uint16_t* active;
  uint8_t* message;
  uint16_t pos;
} sc_ctx;

static void rate_1(sc_ctx* c, uint8_t stage) {
  uint16_t bit_pos = c->active[0];
  uint16_t stage_size = 1U << stage;
  uint8_t* codeword = c->est_bit + bit_pos;
  double* LLR = c->llr0[stage];
  for (uint16_t i = 0; i < stage_size; i++) {
    codeword[i] = (LLR[i] >= 0) ? 0 : 1;
  }
  if (stage == 0) {
    c->message[bit_pos] = codeword[0];
  }
  for (uint8_t i = 0; i <= stage; i++) {
    c->active[i] += (1U << (stage - i));
  }
}

static void rate_0(sc_ctx* c, uint8_t stage) {
  uint16_t bit_pos = c->active[0];
  uint16_t stage_size = 1U << stage;
  uint8_t* codeword = c->est_bit + bit_pos;
  memset(codeword, 0, stage_size);
  if (stage == 0) {
    c->message[bit_pos] = 0;
  }
  for (uint8_t i = 0; i <= stage; i++) {
    c->active[i] += (1U << (stage - i));
  }
}

static void rate_r(sc_ctx* c, uint8_t stage);

static void simplified(sc_ctx* c, uint8_t stage) {
  if (stage == 0) {
    uint16_t bit_pos = c->active[0];
    if (c->frozen[bit_pos]) {
      rate_0(c, 0);
    } else {
      rate_1(c, 0);
    }
    return;
  }
  rate_r(c, stage);
}

static void rate_r(sc_ctx* c, uint8_t stage) {
  uint16_t stage_half_size = 1U << (stage - 1);
  f_vec(c->llr0[stage], c->llr1[stage], c->llr0[stage - 1], stage_half_size);
  simplified(c, stage - 1);
  uint16_t bit_pos = c->active[0];
  int16_t offset0 = bit_pos - stage_half_size;
  uint8_t* estbits0 = c->est_bit + offset0;
  g_vec(estbits0, c->llr0[stage], c->llr1[stage], c->llr0[stage - 1], stage_half_size);
  simplified(c, stage - 1);
  bit_pos = c->active[0];
  offset0 = bit_pos - (1U << stage);
  int16_t offset1 = offset0 + stage_half_size;
  xor_vec(c->est_bit + offset0, c->est_bit + offset1, c->est_bit + offset0, stage_half_size);
  c->active[stage] += 1;
}

void polar_sc_decode(const double* llr, uint8_t* frozen, uint8_t* message, uint8_t n) {
  uint16_t N = 1U << n;
  uint16_t half = N / 2;
  sc_ctx c;
  c.n = n;
  c.N = N;
  c.frozen = frozen;
  c.message = message;
  memset(message, 0, N);

  c.llr0 = (double**)malloc((n + 1) * sizeof(double*));
  c.llr1 = (double**)malloc((n + 1) * sizeof(double*));
  uint16_t total = 1U << (n + 1);
  double* llr_buf = (double*)calloc(total, sizeof(double));
  c.llr0[0] = llr_buf;
  c.llr1[0] = llr_buf + 1;
  for (uint8_t s = 1; s <= n; s++) {
    uint16_t size_s = 1U << s;
    c.llr0[s] = llr_buf + size_s;
    c.llr1[s] = llr_buf + size_s + (1U << (s - 1));
  }
  memcpy(c.llr0[n], llr, half * sizeof(double));
  memcpy(c.llr1[n], llr + half, half * sizeof(double));

  c.est_bit = (uint8_t*)calloc(N, 1);
  c.active = (uint16_t*)calloc(n + 1, sizeof(uint16_t));
  for (uint8_t i = 0; i <= n; i++) {
    c.active[i] = 0;
  }

  simplified(&c, n);

  free(c.active);
  free(c.est_bit);
  free(llr_buf);
  free(c.llr1);
  free(c.llr0);
}
