import math
import random
import numpy as np


def normalize(v):
    ln = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def saturate(val):
    if val > 1.0:
        return 1.0
    elif val < 0.0:
        return 0.0
    else:
        return val


def lerp(val1, val2, weights):
    return val1 * weights + val2 * (1 - weights)


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def ImportanceSampleVisibleGGX(E, Alpha, V):

    Vh = normalize((Alpha[0] * V[0], Alpha[1] * V[1], V[2]))

    Phi = (2 * math.pi) * E[0]
    k = 1.0

    a = saturate(min(Alpha[0], Alpha[1]))
    s = 1.0 + math.sqrt(V[0] ** 2 + V[1] ** 2)
    a2 = a * a
    s2 = s * s
    k = (s2 - a2 * s2) / (s2 + a2 * V[2] * V[2])

    upZ = lerp(1.0, -k * Vh[2], E[1])
    SinTheta = math.sqrt(saturate(1 - upZ * upZ))
    upX = SinTheta * math.cos(Phi)
    upY = SinTheta * math.sin(Phi)
    upH = (upX + Vh[0], upY + Vh[1], upZ + Vh[2])
    upH = normalize((Alpha[0] * upH[0], Alpha[1] * upH[1], max(0.0, upH[2])))

    temp = dot(V, upH)
    o = (
        2.0 * temp * upH[0] - V[0],
        2.0 * temp * upH[1] - V[1],
        2.0 * temp * upH[2] - V[2],
    )
    return o, upH


cnt = 0
SAMPLE = 16384
for roughness in np.arange(0.0, 1.1, 0.1):
    a = roughness**2
    cnt = 0
    for _ in range(SAMPLE):
        alpha = (a, a)
        v = normalize(
            (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        )
        e = (random.uniform(0, 1), random.uniform(0, 1))
        o, m = ImportanceSampleVisibleGGX(e, alpha, v)
        dt = dot(v, m)
        # print(o, dt)
        cnt += 1 if o[2] <= 0 else 0
        # cnt += 1 if dt <= 0 else 0
    print(
        f"roughness: {roughness:.1f} and alpha: {a:.5f} \
        vector o below horizon: {cnt / SAMPLE:.5f}."
    )
