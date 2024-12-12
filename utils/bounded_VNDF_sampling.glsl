float VisibleGGXPDF(float3 V, float3 H, float a2, bool bLimitVDNFToReflection = true)
{
        float NoV = V.z;
        float NoH = H.z;
        float VoH = dot(V, H);

        float d = (NoH * a2 - NoH) * NoH + 1;
        float D = a2 / (PI*d*d);
        float k = 1.0;
#if GGX_BOUNDED_VNDF_SAMPLING
        if (bLimitVDNFToReflection)
        {
                float s = 1.0f + length(V.xy);
                float s2 = s * s;
                k = (s2 - a2 * s2) / (s2 + a2 * V.z * V.z); // Eq. 5
        }
#endif
        float PDF = 2 * VoH * D / (k * NoV + sqrt(NoV * (NoV - NoV * a2) + a2));
        return PDF;
}

float VisibleGGXPDF_aniso(float3 V, float3 H, float2 Alpha, bool bLimitVDNFToReflection = true)
{
        float NoV = V.z;
        float NoH = H.z;
        float VoH = dot(V, H);
        float a2 = Alpha.x * Alpha.y;
        float3 Hs = float3(Alpha.y * H.x, Alpha.x * H.y, a2 * NoH);
        float S = dot(Hs, Hs);
        float D = (1.0f / PI) * a2 * Square(a2 / S);
        float LenV = length(float3(V.x * Alpha.x, V.y * Alpha.y, NoV));
        float k = 1.0;
#if GGX_BOUNDED_VNDF_SAMPLING
        if (bLimitVDNFToReflection)
        {
                float a = saturate(min(Alpha.x, Alpha.y));
                float s = 1.0f + length(V.xy);
                float ka2 = a * a, s2 = s * s;
                k = (s2 - ka2 * s2) / (s2 + ka2 * V.z * V.z); // Eq. 5
        }
#endif
        float Pdf = (2 * D * VoH) / (k * NoV + LenV);
        return Pdf;
}

// PDF = G_SmithV * VoH * D / NoV / (4 * VoH)
// PDF = G_SmithV * D / (4 * NoV)
float4 ImportanceSampleVisibleGGX(float2 E, float2 Alpha, float3 V, bool bLimitVDNFToReflection = true)
{
        // stretch
        float3 Vh = normalize(float3(Alpha * V.xy, V.z));

        // "Sampling Visible GGX Normals with Spherical Caps"
        // Jonathan Dupuy & Anis Benyoub - High Performance Graphics 2023
        float Phi = (2 * PI) * E.x;
        float k = 1.0;
#if GGX_BOUNDED_VNDF_SAMPLING
        if (bLimitVDNFToReflection)
        {
                // If we know we will be reflecting the view vector around the sampled micronormal, we can
                // tweak the range a bit more to eliminate some of the vectors that will point below the horizon
                float a = saturate(min(Alpha.x, Alpha.y));
                float s = 1.0 + length(V.xy);
                float a2 = a * a, s2 = s * s;
                k = (s2 - a2 * s2) / (s2 + a2 * V.z * V.z);
        }
#endif
        float Z = lerp(1.0, -k * Vh.z, E.y);
        float SinTheta = sqrt(saturate(1 - Z * Z));
        float X = SinTheta * cos(Phi);
        float Y = SinTheta * sin(Phi);
        float3 H = float3(X, Y, Z) + Vh;

        // unstretch
        H = normalize(float3(Alpha * H.xy, max(0.0, H.z)));

        return float4(H, VisibleGGXPDF_aniso(V, H, Alpha));
}
