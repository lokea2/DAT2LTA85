/*
 * lth_accel.c  –  C-ускоритель для dat2lta конвертера
 *
 * Функции:
 *   collect_candidates()       — основной поиск доноров (самый горячий цикл)
 *   is_point_inside_triangle() — тест вхождения точки в треугольник
 *   is_coplanar_tri()          — тест компланарности
 *
 * Сборка: см. build_accel.py
 */

#ifdef _WIN32
  #define LTH_EXPORT __declspec(dllexport)
#else
  #define LTH_EXPORT __attribute__((visibility("default")))
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Базовая линейная алгебра                                           */
/* ------------------------------------------------------------------ */

static inline double dot3(const double *a, const double *b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void cross3(const double *a, const double *b, double *out)
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static inline double norm3(const double *v)
{
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

/* ------------------------------------------------------------------ */
/*  is_point_inside_triangle                                           */
/*                                                                     */
/*  p, v0, v1, v2 — указатели на double[3]                            */
/*  Возвращает 1 если внутри (с допуском eps), иначе 0.               */
/* ------------------------------------------------------------------ */
LTH_EXPORT int is_point_inside_triangle(
    const double *p,
    const double *v0, const double *v1, const double *v2,
    double eps)
{
    double e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
    double e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
    double n[3];
    cross3(e1, e2, n);
    double area_sqr = dot3(n, n);
    if (area_sqr < 1e-12) return 0;

    /* барицентрические координаты через скалярные произведения нормалей */
    double b1[3] = { v1[0]-p[0], v1[1]-p[1], v1[2]-p[2] };
    double b2[3] = { v2[0]-p[0], v2[1]-p[1], v2[2]-p[2] };
    double b0[3] = { v0[0]-p[0], v0[1]-p[1], v0[2]-p[2] };

    double c0[3], c1[3], c2[3];
    cross3(b1, b2, c0);
    cross3(b2, b0, c1);
    cross3(b0, b1, c2);

    double u = dot3(n, c0) / area_sqr;
    double v = dot3(n, c1) / area_sqr;
    double w = dot3(n, c2) / area_sqr;

    return (u >= -eps && v >= -eps && w >= -eps) ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/*  is_coplanar_tri                                                    */
/*                                                                     */
/*  n_xyz     — нормаль плоскости double[3]                           */
/*  raw_dist  — дистанция плоскости                                   */
/*  tri_verts — вершины треугольника double[9]  (v0,v1,v2 плоские)    */
/*  Возвращает 1 если все три вершины лежат в плоскости с точностью epsilon */
/* ------------------------------------------------------------------ */
LTH_EXPORT int is_coplanar_tri(
    const double *n_xyz,
    double raw_dist,
    const double *tri_verts,
    double epsilon)
{
    for (int i = 0; i < 3; i++) {
        const double *v = tri_verts + i * 3;
        double d = n_xyz[0]*v[0] + n_xyz[1]*v[1] + n_xyz[2]*v[2] - raw_dist;
        if (d < 0.0) d = -d;
        if (d > epsilon) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Структура результата для одного кандидата                          */
/*  Поля ДОЛЖНЫ совпадать с тем что ожидает Python-обёртка             */
/* ------------------------------------------------------------------ */
typedef struct {
    int32_t orig_idx;
    int32_t matches;
    int32_t winding;
    float   area;
    float   dist;
    float   v_err;
} CandResult;

/* ------------------------------------------------------------------ */
/*  collect_candidates                                                 */
/*                                                                     */
/*  Аргументы (все массивы — row-major, C-порядок):                   */
/*    normals      double[K * 3]      — единичные нормали K треугольников */
/*    centers      double[K * 3]      — центры K треугольников         */
/*    verts        double[K * 3 * 3]  — вершины: K трёуг. × 3 верт. × 3 коорд */
/*    n_arr        double[3]          — нормаль физического полигона   */
/*    raw_dist     double             — дистанция физического полигона */
/*    phys_arr     double[3 * 3]      — 3 физические вершины × 3 коорд */
/*    phys_center  double[3]          — центр физического треугольника */
/*    K            int                — количество render-треугольников */
/*    epsilon      double             — порог компланарности           */
/*    out          CandResult[K]      — выходной буфер (pre-allocated) */
/*                                                                     */
/*  Возвращает количество найденных кандидатов.                        */
/* ------------------------------------------------------------------ */
LTH_EXPORT int collect_candidates(
    const double  *normals,
    const double  *centers,
    const double  *verts,
    const double  *n_arr,
    double         raw_dist,
    const double  *phys_arr,
    const double  *phys_center,
    int            K,
    double         epsilon,
    CandResult    *out)
{
    int count = 0;

    for (int i = 0; i < K; i++) {
        /* ---------- фильтр по нормали (cosine >= 0.5 по абс значению) ---------- */
        const double *ni = normals + i * 3;
        double dot = ni[0]*n_arr[0] + ni[1]*n_arr[1] + ni[2]*n_arr[2];
        double abs_dot = dot < 0.0 ? -dot : dot;
        if (abs_dot < 0.5) continue;

        int winding = (dot > 0.0) ? 1 : 0;

        /* ---------- вершины рендер-треугольника ---------- */
        const double *vi = verts + i * 9;   /* [3 vert][3 coord] */

        /* ---------- matches + v_err ---------- */
        /* matches: кол-во рендер-вершин, которым соответствует хотя бы
           одна физическая вершина (все три оси < 20 одновременно).    */
        /* v_err:   для каждой рендер-вершины — мин. евклидово расст.
           до ближайшей физической вершины; сумма по трём вершинам.    */
        int    matches = 0;
        double v_err   = 0.0;

        for (int rv = 0; rv < 3; rv++) {
            const double *r = vi + rv * 3;
            double min_dist = 1e18;
            int found = 0;

            for (int pv = 0; pv < 3; pv++) {
                const double *p = phys_arr + pv * 3;
                double dx = r[0] - p[0];
                double dy = r[1] - p[1];
                double dz = r[2] - p[2];
                double d  = sqrt(dx*dx + dy*dy + dz*dz);
                if (d < min_dist) min_dist = d;
                if (!found &&
                    (dx < 0 ? -dx : dx) < 20.0 &&
                    (dy < 0 ? -dy : dy) < 20.0 &&
                    (dz < 0 ? -dz : dz) < 20.0)
                {
                    found = 1;
                }
            }
            v_err += min_dist;
            if (found) matches++;
        }

        /* ---------- фильтр компланарности (если matches < 3) ---------- */
        if (matches < 3) {
            if (!is_coplanar_tri(n_arr, raw_dist, vi, epsilon)) continue;
        }

        /* ---------- площадь треугольника ---------- */
        const double *v0 = vi;
        const double *v1 = vi + 3;
        const double *v2 = vi + 6;
        double e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
        double e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
        double cn[3];
        cross3(e1, e2, cn);
        double area = norm3(cn) * 0.5;

        /* ---------- расстояние между центрами ---------- */
        const double *ci = centers + i * 3;
        double ddx = ci[0] - phys_center[0];
        double ddy = ci[1] - phys_center[1];
        double ddz = ci[2] - phys_center[2];
        double dist = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);

        /* ---------- запись результата ---------- */
        out[count].orig_idx = i;
        out[count].matches  = matches;
        out[count].winding  = winding;
        out[count].area     = (float)area;
        out[count].dist     = (float)dist;
        out[count].v_err    = (float)v_err;
        count++;
    }

    return count;
}
