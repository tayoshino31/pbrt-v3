
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// integrators/volpath.cpp*
#include "integrators/nonlinear.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"
#include <random>

namespace pbrt {

Float get_n(Float height){
    Float n0 = 1.000;
    Float dn_dh = 5e-4;
    Float n = n0 + dn_dh * height;
    return n;
}

Float sampleTemperature(Point3f pos) {
    static thread_local std::mt19937 rng(std::random_device{}());
    static thread_local std::uniform_real_distribution<Float> dist(-0.01, 0.01); // ±2°C のノイズ

    Float baseTemp;
    if (pos.y > 1.0)
        baseTemp = 20.0;
    else {
        Float dh = 30;
        baseTemp = 50.0 - dh * pos.y;
    }

    return baseTemp + dist(rng); // ノイズを加える
}

Float computeRefractiveIndex(Float temperatureC, Float pressurePa = 101325.0f) {
    // Convert temperature to Kelvin
    Float T = temperatureC + 273.15f;

    // Standard pressure in Pa
    constexpr Float P0 = 101325.0f;

    // Standard temperature in Kelvin (15°C)
    constexpr Float T0 = 288.15f;

    // Standard refractive index of air at STP (dry air)
    constexpr Float n0 = 1.000293f;

    // Scale with pressure and inverse temperature (ideal gas law approximation)
    return 1.0f +  (n0 - 1.0f) * (pressurePa / P0) * (T0 / T); //TODO remove * 50---------------------------
}

// Float computeRefractiveIndex(Float temperatureC, Float pressurePa = 101325.0f) {
//     const float c1 = 0.0000104f;
//     const float c2 = 0.00366f;
//     Float numerator = c1 * pressurePa * (1.0f + pressurePa * (60.1f - 0.972f * temperatureC) * 1e-10f);
//     Float denominator = 1.0f + c2 * temperatureC;
//     return numerator / denominator;
// }

Vector3f refract(Vector3f d1, Vector3f normal, Float n1, Float n2){
    d1 = Normalize(d1);
    normal  = Normalize(normal);

    Float eta = n1 / n2;
    Float cosTheta1 = Dot(d1, normal);

    // 光が内側から出ていくとき：法線を反転し eta も逆に
    if (cosTheta1 < 0.0f) {
        std::swap(n1, n2);
        eta = n1 / n2;
        normal = -normal;
        cosTheta1 = Dot(d1, normal); 
    }

    Float sinTheta1 = std::sqrt(1.0f - cosTheta1 * cosTheta1);
    Float sintheta2 = eta * sinTheta1;
    Float costheta2 = std::sqrt(1.0f - sintheta2 * sintheta2);
    if(sintheta2 >= 1.0f) return Reflect(-d1, normal); // Total internal reflection
    Vector3f d2 = eta * d1 + (eta * cosTheta1 - costheta2) * normal;
    return Normalize(d2);
}

STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);
STAT_COUNTER("Integrator/Volume interactions", volumeInteractions);
STAT_COUNTER("Integrator/Surface interactions", surfaceInteractions);

// VolPathIntegrator Method Definitions
void NonLinearIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
}

Spectrum NonLinearIntegrator::Li(const RayDifferential &r, const Scene &scene,
                               Sampler &sampler, MemoryArena &arena,
                               int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;

    for (bounces = 0;; ++bounces) {
        // Intersect _ray_ with scene and store intersection in _isect_
        //SurfaceInteraction isect;
        //bool foundIntersection = scene.Intersect(ray, &isect);

        SurfaceInteraction isect;
        Point3f prev_position = ray.o;
        bool foundIntersection;
        Float delta_x = 0.001;
        for (int i_step = 0; i_step < 500; ++i_step){
            foundIntersection = scene.Intersect(ray, &isect);
            if(Distance(prev_position, isect.p) < delta_x)
                break;    
            if (isect.p.x < 0.0f || isect.p.x > 180.0f ||
                isect.p.y < 0.0f || isect.p.y > 0.1f ||
                isect.p.z < -4.0f || isect.p.z > 4.0f) {
                break;
            }
            ray.o += ray.d * delta_x;

            //update ray direction based on refaction
            Float n1 = get_n(prev_position.y);  //computeRefractiveIndex(sampleTemperature(prev_position));
            Float n2 = get_n(ray.o.y); //computeRefractiveIndex(sampleTemperature(ray.o));
            ray.d = refract(ray.d, {0,1,0}, n1, n2);

            prev_position = ray.o;

            //printf("Radiance: %f %f\n", n1, n2);
            //printf("Radiance: %f %f %f\n", prev_position[0], prev_position[1], prev_position[2]);
        }

        // Sample the participating medium, if present
        MediumInteraction mi;
        if (ray.medium) beta *= ray.medium->Sample(ray, sampler, arena, &mi);
        if (beta.IsBlack()) break;

        // Handle an interaction with a medium or a surface
        if (mi.IsValid()) {
            // Terminate path if ray escaped or _maxDepth_ was reached
            if (bounces >= maxDepth) break;

            ++volumeInteractions;
            // Handle scattering at point in medium for volumetric path tracer
            const Distribution1D *lightDistrib =
                lightDistribution->Lookup(mi.p);
            L += beta * UniformSampleOneLight(mi, scene, arena, sampler, true,
                                              lightDistrib);

            Vector3f wo = -ray.d, wi;
            mi.phase->Sample_p(wo, &wi, sampler.Get2D());
            ray = mi.SpawnRay(wi);
            specularBounce = false;
        } else {
            ++surfaceInteractions;
            // Handle scattering at point on surface for volumetric path tracer

            // Possibly add emitted light at intersection
            if (bounces == 0 || specularBounce) {
                // Add emitted light at path vertex or from the environment
                if (foundIntersection)
                    L += beta * isect.Le(-ray.d);
                else
                    for (const auto &light : scene.infiniteLights)
                        L += beta * light->Le(ray);
            }

            // Terminate path if ray escaped or _maxDepth_ was reached
            if (!foundIntersection || bounces >= maxDepth) break;

            // Compute scattering functions and skip over medium boundaries
            isect.ComputeScatteringFunctions(ray, arena, true);
            if (!isect.bsdf) {
                ray = isect.SpawnRay(ray.d);
                bounces--;
                continue;
            }

            // Sample illumination from lights to find attenuated path
            // contribution
            const Distribution1D *lightDistrib =
                lightDistribution->Lookup(isect.p);
            L += beta * UniformSampleOneLight(isect, scene, arena, sampler,
                                              true, lightDistrib);

            // Sample BSDF to get new path direction
            Vector3f wo = -ray.d, wi;
            Float pdf;
            BxDFType flags;
            Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                                              BSDF_ALL, &flags);
            if (f.IsBlack() || pdf == 0.f) break;
            beta *= f * AbsDot(wi, isect.shading.n) / pdf;
            DCHECK(std::isinf(beta.y()) == false);
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
                Float eta = isect.bsdf->eta;
                // Update the term that tracks radiance scaling for refraction
                // depending on whether the ray is entering or leaving the
                // medium.
                etaScale *=
                    (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
            }
            ray = isect.SpawnRay(wi);

            // Account for attenuated subsurface scattering, if applicable
            if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
                // Importance sample the BSSRDF
                SurfaceInteraction pi;
                Spectrum S = isect.bssrdf->Sample_S(
                    scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
                DCHECK(std::isinf(beta.y()) == false);
                if (S.IsBlack() || pdf == 0) break;
                beta *= S / pdf;

                // Account for the attenuated direct subsurface scattering
                // component
                L += beta *
                     UniformSampleOneLight(pi, scene, arena, sampler, true,
                                           lightDistribution->Lookup(pi.p));

                // Account for the indirect subsurface scattering component
                Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(),
                                               &pdf, BSDF_ALL, &flags);
                if (f.IsBlack() || pdf == 0) break;
                beta *= f * AbsDot(wi, pi.shading.n) / pdf;
                DCHECK(std::isinf(beta.y()) == false);
                specularBounce = (flags & BSDF_SPECULAR) != 0;
                ray = pi.SpawnRay(wi);
            }
        }

        // Possibly terminate the path with Russian roulette
        // Factor out radiance scaling due to refraction in rrBeta.
        Spectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(std::isinf(beta.y()) == false);
        }
    }
    ReportValue(pathLength, bounces);
    return L;
}

NonLinearIntegrator *CreateNonLinearIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        params.FindOneString("lightsamplestrategy", "spatial");
    return new NonLinearIntegrator(maxDepth, camera, sampler, pixelBounds,
                                 rrThreshold, lightStrategy);
}

}  // namespace pbrt
