// AVX2 step-response kernel with explicit intrinsics, so the speed does not
// depend on the compiler's auto-vectoriser (MSVC's is weak). Four Hankel
// wavenumbers are processed per instruction; exp and sin/cos use their own
// vectorised polynomials (Cephes-style, ~1 ulp). Each block of four keeps the
// e^-50 attenuation cutoff of evaluateStep in TemSolver.cpp, whose results it
// reproduces to rounding level.
#include "VectorKernel.h"

#include <cmath>
#include <vector>

#if defined(INVERTEM_VECTOR_KERNEL)
#include <immintrin.h>

namespace pytem {
namespace {

constexpr double mu0 = 4.0e-7 * 3.141592653589793238462643383279502884;

using D = __m256d;
struct C {
    D r, i;
};
inline D set(double x) { return _mm256_set1_pd(x); }
inline C add(C a, C b) { return {_mm256_add_pd(a.r, b.r), _mm256_add_pd(a.i, b.i)}; }
inline C sub(C a, C b) { return {_mm256_sub_pd(a.r, b.r), _mm256_sub_pd(a.i, b.i)}; }
inline C scale(D s, C a) { return {_mm256_mul_pd(s, a.r), _mm256_mul_pd(s, a.i)}; }
inline C mul(C a, C b)
{
    return {_mm256_fmsub_pd(a.r, b.r, _mm256_mul_pd(a.i, b.i)),
            _mm256_fmadd_pd(a.r, b.i, _mm256_mul_pd(a.i, b.r))};
}
inline C inv(C a)
{
    const D d = _mm256_div_pd(set(1.0), _mm256_fmadd_pd(a.r, a.r, _mm256_mul_pd(a.i, a.i)));
    return {_mm256_mul_pd(a.r, d), _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), a.i), d)};
}
inline C one() { return {set(1.0), _mm256_setzero_pd()}; }
inline C zero() { return {_mm256_setzero_pd(), _mm256_setzero_pd()}; }

// Round to nearest integer via the 1.5*2^52 trick; returns the rounded
// double and (through bits) the integer in the low mantissa bits.
inline D roundMagic(D x, __m256i &integer)
{
    const D magic = set(6755399441055744.0);
    const D t = _mm256_add_pd(x, magic);
    integer = _mm256_sub_epi64(_mm256_castpd_si256(t), _mm256_castpd_si256(magic));
    return _mm256_sub_pd(t, magic);
}

// exp(x) for x <= 0 (all uses here); 0 below -708.
D expNegative(D x)
{
    const D small = _mm256_cmp_pd(x, set(-708.0), _CMP_LT_OQ);
    x = _mm256_max_pd(x, set(-708.0));
    __m256i n;
    const D k = roundMagic(_mm256_mul_pd(x, set(1.4426950408889634074)), n);
    D r = _mm256_fnmadd_pd(k, set(6.93145751953125e-1), x);
    r = _mm256_fnmadd_pd(k, set(1.42860682030941723212e-6), r);
    // Taylor to r^12 on |r| <= ln2/2: truncation < 2e-16.
    D p = set(1.0 / 479001600.0);
    for (double c : {1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0,
                     1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0})
        p = _mm256_fmadd_pd(p, r, set(c));
    const D scale2n = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_add_epi64(n, _mm256_set1_epi64x(1023)), 52));
    return _mm256_andnot_pd(small, _mm256_mul_pd(p, scale2n));
}

// sin and cos of y with Cody-Waite reduction by pi/2 (accurate for |y| < ~1e6).
void sinCos(D y, D &s, D &c)
{
    __m256i q;
    const D k = roundMagic(_mm256_mul_pd(y, set(0.63661977236758134308)), q);
    D r = _mm256_fnmadd_pd(k, set(1.57079632673412561417e+00), y);
    r = _mm256_fnmadd_pd(k, set(6.07710050630396597660e-11), r);
    r = _mm256_fnmadd_pd(k, set(2.02226624879595063154e-21), r);
    const D z = _mm256_mul_pd(r, r);
    D ps = set(1.58962301576546568060e-10), pc = set(-1.13585365213876817300e-11);
    for (double a : {-2.50507477628578072866e-8, 2.75573136213857245213e-6, -1.98412698295895385996e-4,
                     8.33333333332211858878e-3, -1.66666666666666307295e-1})
        ps = _mm256_fmadd_pd(ps, z, set(a));
    for (double a : {2.08757008419747316778e-9, -2.75573141792967388112e-7, 2.48015872888517045348e-5,
                     -1.38888888888730564116e-3, 4.16666666666665929218e-2})
        pc = _mm256_fmadd_pd(pc, z, set(a));
    const D sinR = _mm256_fmadd_pd(_mm256_mul_pd(r, z), ps, r);
    const D cosR = _mm256_fmadd_pd(_mm256_mul_pd(z, z), pc, _mm256_fnmadd_pd(set(0.5), z, set(1.0)));
    // Quadrant q: odd swaps sin/cos; q&2 negates sin; (q+1)&2 negates cos.
    const D odd = _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(q, _mm256_set1_epi64x(1)),
                                                         _mm256_set1_epi64x(1)));
    const D sign = set(-0.0);
    const D flipSin = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_and_si256(q, _mm256_set1_epi64x(2)), 62));
    const D flipCos = _mm256_castsi256_pd(_mm256_slli_epi64(
        _mm256_and_si256(_mm256_add_epi64(q, _mm256_set1_epi64x(1)), _mm256_set1_epi64x(2)), 62));
    s = _mm256_xor_pd(_mm256_blendv_pd(sinR, cosR, odd), _mm256_and_pd(flipSin, sign));
    c = _mm256_xor_pd(_mm256_blendv_pd(cosR, sinR, odd), _mm256_and_pd(flipCos, sign));
}

// Principal sqrt for Re(z) > 0: |z| via sqrt(a^2 + b^2).
inline C csqrt(C z)
{
    const D m = _mm256_sqrt_pd(_mm256_fmadd_pd(z.r, z.r, _mm256_mul_pd(z.i, z.i)));
    const D re = _mm256_sqrt_pd(_mm256_mul_pd(set(0.5), _mm256_add_pd(m, z.r)));
    return {re, _mm256_div_pd(_mm256_mul_pd(set(0.5), z.i), re)};
}

inline C cexp(C z)
{
    D s, c;
    sinCos(z.i, s, c);
    const D e = expNegative(z.r);
    return {_mm256_mul_pd(e, c), _mm256_mul_pd(e, s)};
}

inline double sum4(D x)
{
    alignas(32) double v[4];
    _mm256_store_pd(v, x);
    return (v[0] + v[1]) + (v[2] + v[3]);
}

} // namespace

void vectorKernelTime(const KernelView &v, std::size_t t, double &response, double *jacobian)
{
    const std::size_t n = v.layers, L = v.lambdaCount, B = (L + 3) / 4;
    // Pad the wavenumbers to a multiple of four with zero-weight copies.
    std::vector<double> lam(4 * B, v.lambdas[L - 1]), lam2(4 * B, v.lambdaSquared[L - 1]), hf(4 * B, 0.0);
    for (std::size_t l = 0; l < L; ++l) {
        lam[l] = v.lambdas[l];
        lam2[l] = v.lambdaSquared[l];
        hf[l] = v.hankelFactors[l];
    }
    std::vector<C> G(n), E(n), P(n), W(n), Q(n), F(n), T(n);
    response = 0.0;
    for (std::size_t f = 0; f < v.frequencies; ++f) {
        const std::complex<double> s = v.s[t * v.frequencies + f];
        for (std::size_t j = 0; j < n; ++j)
            T[j] = {set(s.real() * mu0 / v.resistivities[j]), set(s.imag() * mu0 / v.resistivities[j])};
        C total = zero();
        if (jacobian)
            std::fill(F.begin(), F.end(), zero());
        for (std::size_t b = 0; b < B; ++b) {
            const C lambda{_mm256_loadu_pd(&lam[4 * b]), _mm256_setzero_pd()};
            const D lambda2 = _mm256_loadu_pd(&lam2[4 * b]);
            const D weight = _mm256_loadu_pd(&hf[4 * b]);
            // Top-down: gamma, decay and the per-lane e^-50 attenuation cutoff.
            std::size_t depth = n;
            D att = _mm256_setzero_pd();
            for (std::size_t j = 0; j < n; ++j) {
                G[j] = v.sameRho[j] ? G[j - 1] : csqrt({_mm256_add_pd(lambda2, T[j].r), T[j].i});
                if (j + 1 == n) {
                    E[j] = zero();
                    break;
                }
                const D h2 = set(2.0 * v.thicknesses[j]);
                E[j] = v.sameLayer[j] ? E[j - 1] : cexp(scale(set(-2.0 * v.thicknesses[j]), G[j]));
                att = _mm256_fmadd_pd(h2, G[j].r, att);
                const D cut = _mm256_cmp_pd(att, set(50.0), _CMP_GT_OQ);
                E[j] = {_mm256_andnot_pd(cut, E[j].r), _mm256_andnot_pd(cut, E[j].i)};
                if (_mm256_movemask_pd(cut) == 15) {
                    depth = j + 1;
                    break;
                }
            }
            // Bottom-up reflection recursion.
            C R = zero();
            for (std::size_t j = depth; j-- > 0;) {
                W[j] = R;
                if (v.sameRho[j]) { // no interface: psi = 0
                    P[j] = zero();
                    R = mul(R, E[j]);
                    continue;
                }
                const C above = j == 0 ? lambda : G[j - 1];
                P[j] = mul(sub(above, G[j]), inv(add(above, G[j])));
                const C rd = mul(R, E[j]);
                R = mul(add(P[j], rd), inv(add(one(), mul(P[j], rd))));
            }
            total = add(total, scale(weight, R));
            if (!jacobian)
                continue;
            // Adjoint of the recursion, top-down: d(total)/d(ln rho_j).
            C A{weight, _mm256_setzero_pd()};
            for (std::size_t j = 0; j < depth; ++j) {
                const C invG = inv(G[j]);
                Q[j] = v.sameRho[j] ? Q[j - 1] : mul(scale(set(-0.5), T[j]), invG);
                const C dDecay = scale(set(j + 1 < n ? -2.0 * v.thicknesses[j] : 0.0), E[j]);
                const C bd = mul(W[j], E[j]);
                if (v.sameRho[j]) { // psi = 0, above = gamma: 1/(above+gamma)^2 -> 1/(2 gamma)
                    const C q = mul(mul(scale(set(0.5), invG), A), mul(sub(one(), mul(bd, bd)), Q[j]));
                    F[j] = add(F[j], sub(mul(mul(A, W[j]), mul(dDecay, Q[j])), q));
                    F[j - 1] = add(F[j - 1], q);
                    A = mul(A, E[j]);
                    continue;
                }
                const C above = j == 0 ? lambda : G[j - 1];
                const C den = add(one(), mul(P[j], bd));
                const C inv2 = inv(mul(den, den));
                const C dPsi = mul(sub(one(), mul(bd, bd)), inv2);
                const C sumG = add(above, G[j]);
                const C invSum2 = inv(mul(sumG, sumG));
                const C onePsi2 = mul(sub(one(), mul(P[j], P[j])), inv2);
                F[j] = add(F[j], mul(mul(A, add(mul(scale(set(-2.0), above), mul(invSum2, dPsi)),
                                                mul(W[j], mul(onePsi2, dDecay)))), Q[j]));
                if (j > 0)
                    F[j - 1] = add(F[j - 1], mul(mul(A, dPsi), mul(scale(set(2.0), mul(G[j], invSum2)), Q[j - 1])));
                A = mul(A, mul(E[j], onePsi2));
            }
        }
        const std::complex<double> c = v.coefficients[t * v.frequencies + f];
        response += c.real() * sum4(total.r) - c.imag() * sum4(total.i);
        if (jacobian)
            for (std::size_t j = 0; j < n; ++j)
                jacobian[j] += c.real() * sum4(F[j].r) - c.imag() * sum4(F[j].i);
    }
}

} // namespace pytem

#else

namespace pytem {
void vectorKernelTime(const KernelView &, std::size_t, double &response, double *)
{
    response = 0.0;
}
} // namespace pytem

#endif
