/* worstcase.c -- C implementation of the box-support W1 worst-case CVaR value.
 *
 * Mirrors portfolio/utils.py::worst_case_value_box (Sections 9.5 + 9.9 of
 * port_box/METHOD.md) exactly:
 *
 *   outer: 60-iteration bisection on lambda in (0, ||a x||_2] for the root of
 *          h'(lam) = eps - sum_i w_i r_i [active_i]  (h convex, h' nondecreasing);
 *   inner: per sample, the piece-2 box-constrained concave maximization
 *          sup_{zeta in [lb,ub]} b^T zeta + c2 - lam ||zeta - xi_i||_2
 *          via the clipped step zeta*(mu) = clip(xi + b/mu, lb, ub) and the
 *          self-consistency condition  mu ||zeta*(mu) - xi|| = lam.
 *
 * Two interchangeable inner solvers (inner_mode):
 *   0 = inner_bisect: the literal 80-iteration mu-bisection of the numpy code
 *       (same brackets, same clip/update, same final midpoint evaluation);
 *       ground-truth-parity path.  Compile with -ffp-contract=off so the
 *       bisection sign tests see the same roundings as numpy.
 *   1 = exact: exact piecewise closed form of the SAME optimality condition.
 *       With t = 1/mu, zeta_j(t) = clip(xi_j + t b_j) is piecewise linear in
 *       t, so between the coordinate saturation breakpoints
 *           r(t)^2      = A t^2 + B,
 *           b.zeta(t)   = C + A t,
 *       with A = sum b_j^2 over linear coords and B, C collecting the clipped
 *       coords.  The breakpoints and the per-piece constants (A_k, B_k, C_k)
 *       do NOT depend on lambda, so they are precomputed once per call; each
 *       outer evaluation then solves r(t)/t = lam per sample with one
 *       binary search over the pieces (q(t) = A + B/t^2 is continuous and
 *       nonincreasing; the root on a piece is t* = sqrt(B/(lam^2 - A))) and
 *       O(1) arithmetic.  The resulting multiplier is clamped to the
 *       bisection bracket [mu_lo, mu_hi] so the degenerate no-root cases
 *       (e.g. samples sitting exactly on box faces) reproduce the
 *       bisection's limit behaviour.
 *
 * Pure C99, no external dependencies.  Double precision throughout.  Optional
 * OpenMP over samples (production runs use 1 thread).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */
static double clip1(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

typedef struct { double t, dA, dB, dC; } wc_event;

static int wc_event_cmp(const void *pa, const void *pb)
{
    double ta = ((const wc_event *)pa)->t;
    double tb = ((const wc_event *)pb)->t;
    return (ta > tb) - (ta < tb);
}

/* mu-bracket of the numpy bisection (per sample) */
static void mu_bracket(double lam, double nb, double rmax,
                       double *mu_lo, double *mu_hi)
{
    double lo = lam / (rmax > 1e-12 ? rmax : 1e-12);
    double hi = nb / 1e-9;
    if (hi < lo) hi = lo;                 /* guard atoms already at the corner */
    *mu_lo = lo;
    *mu_hi = hi;
}

/* clip evaluation at a given mu: fills zeta, returns r and b.zeta */
static void eval_mu(const double *xi, const double *b, int d, double mu,
                    const double *lb, const double *ub,
                    double *zeta, double *r_out, double *bz_out)
{
    double r2 = 0.0, bz = 0.0;
    for (int j = 0; j < d; ++j) {
        double z = clip1(xi[j] + b[j] / mu, lb[j], ub[j]);
        zeta[j] = z;
        double dv = z - xi[j];
        r2 += dv * dv;
        bz += b[j] * z;
    }
    *r_out = sqrt(r2);
    *bz_out = bz;
}

/* ------------------------------------------------------------------ */
/* inner solver a: literal 80-iteration mu-bisection (numpy parity)   */
/* ------------------------------------------------------------------ */
static void inner_bisect(const double *xi, const double *b, int d,
                         double lam, double nb,
                         const double *lb, const double *ub,
                         double rmax, int mu_iter,
                         double *zeta, double *r_out, double *bz_out)
{
    double mu_lo, mu_hi;
    mu_bracket(lam, nb, rmax, &mu_lo, &mu_hi);
    for (int it = 0; it < mu_iter; ++it) {
        double mu = 0.5 * (mu_lo + mu_hi);
        double r2 = 0.0;
        for (int j = 0; j < d; ++j) {
            double dv = clip1(xi[j] + b[j] / mu, lb[j], ub[j]) - xi[j];
            r2 += dv * dv;
        }
        if (mu * sqrt(r2) - lam > 0.0) mu_hi = mu; else mu_lo = mu;
    }
    eval_mu(xi, b, d, 0.5 * (mu_lo + mu_hi), lb, ub, zeta, r_out, bz_out);
}

/* ------------------------------------------------------------------ */
/* inner solver b: exact piecewise closed form, lam-independent part  */
/* ------------------------------------------------------------------ */
/* Per-sample piece tables.  Sample i has nev[i] breakpoints
 * tb[0] < ... < tb[nev-1] (stride 2d) and nev[i]+1 pieces with constants
 * A[k], B[k], C[k] (stride 2d+1): on piece k (t between tb[k-1] and tb[k])
 * r(t)^2 = A_k t^2 + B_k and b.zeta(t) = C_k + A_k t.                     */
typedef struct {
    double *tb, *A, *B, *C;
    int *nev;
    int stride_tb, stride_p;
} exact_pre;

static int exact_precompute_row(const double *xi, const double *b, int d,
                                const double *lb, const double *ub,
                                wc_event *ev,
                                double *tb, double *A, double *B, double *C)
{
    double A0 = 0.0, B0 = 0.0, C0 = 0.0;
    int nev = 0;
    for (int j = 0; j < d; ++j) {
        double bj = b[j];
        if (bj == 0.0) {                       /* never moves; b.zeta term 0 */
            double dv = clip1(xi[j], lb[j], ub[j]) - xi[j];
            B0 += dv * dv;
            continue;
        }
        double s = (bj > 0.0) ? ub[j] : lb[j]; /* favorable bound */
        double o = (bj > 0.0) ? lb[j] : ub[j]; /* opposite bound  */
        double t_out = (s - xi[j]) / bj;
        if (t_out <= 0.0) {                    /* saturated from the start */
            double dv = s - xi[j];
            B0 += dv * dv;
            C0 += bj * s;
            continue;
        }
        double t_in = (o - xi[j]) / bj;
        if (t_in > 0.0) {                      /* starts clipped at o_j */
            double dv = o - xi[j];
            B0 += dv * dv;
            C0 += bj * o;
            ev[nev].t = t_in; ev[nev].dA = bj * bj; ev[nev].dB = -dv * dv;
            ev[nev].dC = bj * xi[j] - bj * o;
            ++nev;
        } else {                               /* starts in the linear phase */
            A0 += bj * bj;
            C0 += bj * xi[j];
        }
        double dvs = s - xi[j];
        ev[nev].t = t_out; ev[nev].dA = -(bj * bj); ev[nev].dB = dvs * dvs;
        ev[nev].dC = bj * s - bj * xi[j];
        ++nev;
    }
    qsort(ev, (size_t)nev, sizeof(wc_event), wc_event_cmp);
    A[0] = A0; B[0] = B0; C[0] = C0;
    for (int k = 0; k < nev; ++k) {
        tb[k] = ev[k].t;
        A[k + 1] = A[k] + ev[k].dA;
        B[k + 1] = B[k] + ev[k].dB;
        C[k + 1] = C[k] + ev[k].dC;
        if (A[k + 1] < 0.0) A[k + 1] = 0.0;    /* fp accumulation guards */
        if (B[k + 1] < 0.0) B[k + 1] = 0.0;
    }
    /* overwrite the final (corner) piece with directly computed constants to
     * kill accumulated rounding: zeta = favorable corner / clipped xi.      */
    double Bf = 0.0, Cf = 0.0;
    for (int j = 0; j < d; ++j) {
        double bj = b[j];
        if (bj == 0.0) {
            double dv = clip1(xi[j], lb[j], ub[j]) - xi[j];
            Bf += dv * dv;
        } else {
            double s = (bj > 0.0) ? ub[j] : lb[j];
            double dv = s - xi[j];
            Bf += dv * dv;
            Cf += bj * s;
        }
    }
    A[nev] = 0.0; B[nev] = Bf; C[nev] = Cf;
    return nev;
}

/* Solve r(t)/t = lam for sample row (tb, A, B, C, nev); return the clamped
 * t and the index of the piece containing it.                              */
static void exact_solve_row(const double *tb, const double *A,
                            const double *B, const double *C, int nev,
                            double lam, double mu_lo, double mu_hi,
                            double *t_out, int *piece_out)
{
    (void)C;
    double lam2 = lam * lam;
    /* first piece k with q(end of k) <= lam2 (q nonincreasing); default nev */
    int lo = 0, hi = nev;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        double q = A[mid] + B[mid] / (tb[mid] * tb[mid]);
        if (q <= lam2) hi = mid; else lo = mid + 1;
    }
    int k = lo;
    double t;
    if (B[k] > 0.0) {
        double den = lam2 - A[k];
        t = (den > 0.0) ? sqrt(B[k] / den) : HUGE_VAL;
    } else {
        /* all-linear piece: plateau; take its start (0 for the first piece) */
        t = (k > 0) ? tb[k - 1] : 0.0;
    }
    /* clamp to the numpy bisection bracket (t = 1/mu) */
    double t_min = 1.0 / mu_hi, t_max = 1.0 / mu_lo;
    if (t < t_min) t = t_min;
    if (t > t_max) t = t_max;
    /* relocate the piece containing t (clamping may have moved it) */
    lo = 0; hi = nev;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (tb[mid] <= t) lo = mid + 1; else hi = mid;
    }
    *t_out = t;
    *piece_out = lo;
}

/* ------------------------------------------------------------------ */
/* Kernel A: one outer-dual evaluation h(lam), h'(lam)                */
/* ------------------------------------------------------------------ */
/* Computes hval = lam*eps + sum_i w_i max(phi_i, 0) and
 * hprime = eps - sum_i w_i r_i [phi_i > 0].  When zeta_out / active_out are
 * non-NULL also writes the per-sample maximizers (N x d, row-major) and the
 * active flags.  Row-cache-resident: loop over samples outer, coordinates
 * inner.  ``pre`` may be NULL for inner_mode 0.                            */
void wc_eval_lam(const double *dat, long N, int d,
                 const double *b, double c2, double lam, double nb,
                 const double *lb, const double *ub,
                 const double *zbar, double bzbar, const double *rmax,
                 const double *w, double eps,
                 int inner_mode, int mu_iter, const exact_pre *pre,
                 double *hval_out, double *hprime_out,
                 double *zeta_out, unsigned char *active_out)
{
    double acc_phi = 0.0, acc_r = 0.0;

    if (nb == 0.0 || lam >= nb) {
        /* Cauchy-Schwarz: moving never beats the transport cost -> stay */
        for (long i = 0; i < N; ++i) {
            const double *xi = dat + (size_t)i * d;
            double bz = 0.0;
            for (int j = 0; j < d; ++j) bz += b[j] * xi[j];
            double phi = bz + c2;
            int act = phi > 0.0;
            if (act) acc_phi += w[i] * phi;    /* r_i = 0 */
            if (zeta_out) memcpy(zeta_out + (size_t)i * d, xi,
                                 (size_t)d * sizeof(double));
            if (active_out) active_out[i] = (unsigned char)act;
        }
    } else if (lam <= 1e-15) {
        /* zero transport cost: jump straight to the favorable corner */
        for (long i = 0; i < N; ++i) {
            double phi = bzbar + c2 - lam * rmax[i];
            int act = phi > 0.0;
            if (act) { acc_phi += w[i] * phi; acc_r += w[i] * rmax[i]; }
            if (zeta_out) memcpy(zeta_out + (size_t)i * d, zbar,
                                 (size_t)d * sizeof(double));
            if (active_out) active_out[i] = (unsigned char)act;
        }
    } else if (inner_mode != 0 && pre != NULL) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:acc_phi, acc_r)
#endif
        for (long i = 0; i < N; ++i) {
            const double *xi = dat + (size_t)i * d;
            const double *tb = pre->tb + (size_t)i * pre->stride_tb;
            const double *A = pre->A + (size_t)i * pre->stride_p;
            const double *B = pre->B + (size_t)i * pre->stride_p;
            const double *C = pre->C + (size_t)i * pre->stride_p;
            double mu_lo, mu_hi, t, r, bz;
            int k;
            mu_bracket(lam, nb, rmax[i], &mu_lo, &mu_hi);
            exact_solve_row(tb, A, B, C, pre->nev[i], lam, mu_lo, mu_hi,
                            &t, &k);
            if (zeta_out) {
                /* final evaluation: materialize zeta via the clip formula
                 * (matches the numpy final-midpoint evaluation semantics) */
                eval_mu(xi, b, d, 1.0 / t, lb, ub,
                        zeta_out + (size_t)i * d, &r, &bz);
            } else {
                r = sqrt(A[k] * t * t + B[k]);
                bz = C[k] + A[k] * t;
            }
            double phi = bz + c2 - lam * r;
            int act = phi > 0.0;
            if (act) { acc_phi += w[i] * phi; acc_r += w[i] * r; }
            if (active_out) active_out[i] = (unsigned char)act;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:acc_phi, acc_r)
#endif
        for (long i = 0; i < N; ++i) {
            const double *xi = dat + (size_t)i * d;
            double zeta_local[d ? d : 1];      /* C99 VLA, row-sized */
            double *zrow = zeta_out ? zeta_out + (size_t)i * d : zeta_local;
            double r, bz;
            inner_bisect(xi, b, d, lam, nb, lb, ub, rmax[i], mu_iter,
                         zrow, &r, &bz);
            double phi = bz + c2 - lam * r;
            int act = phi > 0.0;
            if (act) { acc_phi += w[i] * phi; acc_r += w[i] * r; }
            if (active_out) active_out[i] = (unsigned char)act;
        }
    }
    *hval_out = lam * eps + acc_phi;
    *hprime_out = eps - acc_r;
}

/* ------------------------------------------------------------------ */
/* Kernel B: full worst-case value (outer lambda bisection)           */
/* ------------------------------------------------------------------ */
/* Identical control flow to the Python worst_case_value_box:
 *   - eps <= 0 or ||b|| == 0: empirical value, lam* = 0, zeta = dat;
 *   - otherwise evaluate h'(0); if >= 0, lam* = 0; else 60-iteration
 *     bisection on (0, ||b||], final evaluation at the bracket midpoint.
 * zeta_out (N x d) and active_out (N) may be NULL when the caller does not
 * need the subgradient state.  Returns 0 on success, -1 on alloc failure.  */
int worstcase_box(const double *dat, long N, int d,
                  const double *x, double tau, const double *w,
                  double eps, const double *lb, const double *ub, double a,
                  int lam_iter, int mu_iter, int inner_mode,
                  double *F_out, double *lam_out,
                  double *zeta_out, unsigned char *active_out)
{
    double *b = (double *)malloc((size_t)d * sizeof(double));
    double *zbar = (double *)malloc((size_t)d * sizeof(double));
    double *rmax = (double *)malloc((size_t)N * sizeof(double));
    if (!b || !zbar || !rmax) { free(b); free(zbar); free(rmax); return -1; }

    double c2 = a * tau;
    double nb2 = 0.0;
    for (int j = 0; j < d; ++j) { b[j] = a * x[j]; nb2 += b[j] * b[j]; }
    double nb = sqrt(nb2);

    if (eps <= 0.0 || nb == 0.0) {            /* empirical (non-robust) value */
        double acc = 0.0;
        for (long i = 0; i < N; ++i) {
            const double *xi = dat + (size_t)i * d;
            double bz = 0.0;
            for (int j = 0; j < d; ++j) bz += b[j] * xi[j];
            double phi = bz + c2;
            int act = phi > 0.0;
            if (act) acc += w[i] * phi;
            if (zeta_out) memcpy(zeta_out + (size_t)i * d, xi,
                                 (size_t)d * sizeof(double));
            if (active_out) active_out[i] = (unsigned char)act;
        }
        *F_out = tau + acc;
        *lam_out = 0.0;
        free(b); free(zbar); free(rmax);
        return 0;
    }

    double bzbar = 0.0;
    for (int j = 0; j < d; ++j) {
        zbar[j] = (b[j] > 0.0) ? ub[j] : lb[j];    /* gradient-favorable corner */
        bzbar += b[j] * zbar[j];
    }
    for (long i = 0; i < N; ++i) {
        const double *xi = dat + (size_t)i * d;
        double r2 = 0.0;
        for (int j = 0; j < d; ++j) {
            double dv = zbar[j] - xi[j];
            r2 += dv * dv;
        }
        rmax[i] = sqrt(r2);
    }

    /* lam-independent precompute for the exact inner solver */
    exact_pre pre;
    exact_pre *pre_p = NULL;
    memset(&pre, 0, sizeof(pre));
    if (inner_mode != 0) {
        pre.stride_tb = 2 * d;
        pre.stride_p = 2 * d + 1;
        pre.tb = (double *)malloc((size_t)N * pre.stride_tb * sizeof(double));
        pre.A = (double *)malloc((size_t)N * pre.stride_p * sizeof(double));
        pre.B = (double *)malloc((size_t)N * pre.stride_p * sizeof(double));
        pre.C = (double *)malloc((size_t)N * pre.stride_p * sizeof(double));
        pre.nev = (int *)malloc((size_t)N * sizeof(int));
        if (!pre.tb || !pre.A || !pre.B || !pre.C || !pre.nev) {
            free(pre.tb); free(pre.A); free(pre.B); free(pre.C); free(pre.nev);
            free(b); free(zbar); free(rmax);
            return -1;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < N; ++i) {
            wc_event ev_local[d > 0 ? 2 * d : 1];  /* C99 VLA scratch */
            pre.nev[i] = exact_precompute_row(
                dat + (size_t)i * d, b, d, lb, ub, ev_local,
                pre.tb + (size_t)i * pre.stride_tb,
                pre.A + (size_t)i * pre.stride_p,
                pre.B + (size_t)i * pre.stride_p,
                pre.C + (size_t)i * pre.stride_p);
        }
        pre_p = &pre;
    }

    double hval, hprime, lam_star;
    /* h convex on [0, ||b||]; h' nondecreasing.  h'(||b||) = eps > 0, so a
     * root sits in [0, ||b||) unless the budget already exceeds the box
     * diameter (then lam* = 0).                                            */
    wc_eval_lam(dat, N, d, b, c2, 0.0, nb, lb, ub, zbar, bzbar, rmax,
                w, eps, inner_mode, mu_iter, pre_p, &hval, &hprime,
                NULL, NULL);
    if (hprime >= 0.0) {
        lam_star = 0.0;
    } else {
        double lo = 0.0, hi = nb;
        for (int it = 0; it < lam_iter; ++it) {
            double mid = 0.5 * (lo + hi);
            wc_eval_lam(dat, N, d, b, c2, mid, nb, lb, ub, zbar, bzbar, rmax,
                        w, eps, inner_mode, mu_iter, pre_p, &hval, &hprime,
                        NULL, NULL);
            if (hprime > 0.0) hi = mid; else lo = mid;
        }
        lam_star = 0.5 * (lo + hi);
    }

    /* Final evaluation.  When the caller wants the subgradient state, use
     * the literal mu-bisection regardless of inner_mode: at a kink-pinned
     * lam* the inner maximizer can be non-unique (plateau), and the numpy
     * bisection's selection behaviour -- which the downstream Danskin
     * subgradient step is tuned to -- is reproduced by the bisect path,
     * whereas the exact solver would deterministically pick one plateau end.
     * Value-only calls keep the fast exact final evaluation (F differs only
     * at the ~1e-12 level between the two).                                */
    int final_mode = (zeta_out != NULL) ? 0 : inner_mode;
    wc_eval_lam(dat, N, d, b, c2, lam_star, nb, lb, ub, zbar, bzbar, rmax,
                w, eps, final_mode, mu_iter, pre_p, &hval, &hprime,
                zeta_out, active_out);
    *F_out = tau + hval;
    *lam_out = lam_star;
    if (pre_p) {
        free(pre.tb); free(pre.A); free(pre.B); free(pre.C); free(pre.nev);
    }
    free(b); free(zbar); free(rmax);
    return 0;
}
