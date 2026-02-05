import * as THREE from 'three';

function toColor(value, fallback) {
    if (value instanceof THREE.Color) {
        return value.clone();
    }
    if (typeof value === 'number' || typeof value === 'string') {
        return new THREE.Color(value);
    }
    if (value && typeof value === 'object') {
        const c = new THREE.Color();
        c.set(value);
        return c;
    }
    return fallback instanceof THREE.Color ? fallback.clone() : new THREE.Color(fallback ?? 0xffffff);
}

function resolveDimensions(dimensions) {
    if (dimensions instanceof THREE.Vector3) {
        return dimensions.clone();
    }
    const d = new THREE.Vector3(1, 1, 1);
    if (!dimensions || typeof dimensions !== 'object') {
        return d;
    }
    const x = dimensions.x ?? dimensions.width ?? dimensions.w;
    const y = dimensions.y ?? dimensions.height ?? dimensions.h;
    const z = dimensions.z ?? dimensions.depth ?? dimensions.d;
    if (typeof x === 'number' && Number.isFinite(x)) d.x = Math.max(Math.abs(x), 1e-3);
    if (typeof y === 'number' && Number.isFinite(y)) d.y = Math.max(Math.abs(y), 1e-3);
    if (typeof z === 'number' && Number.isFinite(z)) d.z = Math.max(Math.abs(z), 1e-3);
    return d;
}

/**
 * Create a MeshPhysicalMaterial configured with a custom shader that
 * introduces holographic gradients, animated-looking stripes and rim glows.
 * The shader runs entirely on the GPU and can be recoloured at runtime via
 * {@link updateSciFiMaterialColor}.
 */
export function createSciFiMaterial(options = {}) {
    const {
        baseColor = 0x041021,
        accentColor = 0x4cd8ff,
        secondaryColor = 0x050d1a,
        edgeColor = 0xa5f6ff,
        emissiveColor = 0x53e2ff,
        emissiveIntensity = 0.48,
        metalness = 0.78,
        roughness = 0.12,
        clearcoat = 0.92,
        clearcoatRoughness = 0.18,
        transmission = 0.16,
        thickness = 1.6,
        iridescence = 0.55,
        iridescenceIOR = 1.32,
        sheen = 0.55,
        sheenColor = 0xb9ffff,
        sheenRoughness = 0.42,
        envMapIntensity = 1.8,
        transparent = true,
        opacity = 0.88,
        side = THREE.FrontSide,
        dimensions = new THREE.Vector3(1, 1, 1),
        stripeFrequency = 8.0,
        stripeStrength = 0.45,
        rimIntensity = 0.72,
        gradientSharpness = 1.55,
        gradientBias = 0.035,
        fresnelBoost = 0.42,
        accentMix = 0.9,
        glowFalloff = 1.9,
        depthAccentStrength = 0.25,
        scanlineFrequency = 14.0,
        scanlineStrength = 0.2,
        glintStrength = 0.16,
        noiseStrength = 0.035,
        extraUniforms = {}
    } = options;

    const material = new THREE.MeshPhysicalMaterial({
        color: toColor(baseColor, 0x060913),
        metalness,
        roughness,
        clearcoat,
        clearcoatRoughness,
        transmission,
        thickness,
        iridescence,
        iridescenceIOR,
        sheen,
        sheenColor: toColor(sheenColor, 0x99ffff),
        sheenRoughness,
        envMapIntensity,
        transparent,
        opacity,
        side,
        emissive: toColor(emissiveColor, 0x2fffff),
        emissiveIntensity
    });
    material.extensions = material.extensions || {};
    material.extensions.derivatives = true;

    const dims = resolveDimensions(dimensions);

    const uniforms = {
        uAccentColor: { value: toColor(accentColor, 0x33e0ff) },
        uSecondaryColor: { value: toColor(secondaryColor, 0x09111f) },
        uEdgeColor: { value: toColor(edgeColor, 0x7fffff) },
        uDimensions: { value: dims },
        uStripeFrequency: { value: stripeFrequency },
        uStripeStrength: { value: stripeStrength },
        uRimIntensity: { value: rimIntensity },
        uGradientSharpness: { value: gradientSharpness },
        uGradientBias: { value: gradientBias },
        uFresnelBoost: { value: fresnelBoost },
        uAccentMix: { value: accentMix },
        uGlowFalloff: { value: glowFalloff },
        uDepthAccentStrength: { value: depthAccentStrength },
        uScanlineFrequency: { value: scanlineFrequency },
        uScanlineStrength: { value: scanlineStrength },
        uGlintStrength: { value: glintStrength },
        uNoiseStrength: { value: noiseStrength }
    };

    for (const [key, value] of Object.entries(extraUniforms)) {
        uniforms[key] = value;
    }

    material.userData.sciFiUniforms = uniforms;

    material.onBeforeCompile = (shader) => {
        shader.uniforms = { ...shader.uniforms, ...uniforms };

        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', '#include <common>\nvarying vec3 vLocalPos;\nvarying vec3 vWorldNormal;\n')
            .replace(
                '#include <begin_vertex>',
                '#include <begin_vertex>\n    #ifdef USE_INSTANCING\n        vec4 instPos = instanceMatrix * vec4(position, 1.0);\n        vLocalPos = instPos.xyz;\n    #else\n        vLocalPos = position;\n    #endif\n    vWorldNormal = normalize(normalMatrix * normal);\n'
            );

        shader.fragmentShader = shader.fragmentShader
            .replace(
                '#include <common>',
                '#include <common>\nvarying vec3 vLocalPos;\nvarying vec3 vWorldNormal;\nuniform vec3 uAccentColor;\nuniform vec3 uSecondaryColor;\nuniform vec3 uEdgeColor;\nuniform vec3 uDimensions;\nuniform float uStripeFrequency;\nuniform float uStripeStrength;\nuniform float uRimIntensity;\nuniform float uGradientSharpness;\nuniform float uGradientBias;\nuniform float uFresnelBoost;\nuniform float uAccentMix;\nuniform float uGlowFalloff;\nuniform float uDepthAccentStrength;\nuniform float uScanlineFrequency;\nuniform float uScanlineStrength;\nuniform float uGlintStrength;\nuniform float uNoiseStrength;\n'
            )
            .replace(
                '#include <normal_fragment_begin>',
                '#include <normal_fragment_begin>\nfloat topMask = smoothstep(0.6, 0.92, abs(normalize(vWorldNormal).y));\n'
            )
            .replace(
                'float roughnessFactor = roughness;',
                'float roughnessFactor = mix(roughness, max(roughness, 0.65), topMask);'
            )
            .replace(
                'float clearcoat = clearcoat;',
                'float clearcoat = mix(clearcoat, clearcoat * 0.35, topMask);'
            )
            .replace(
                'float clearcoatRoughness = clearcoatRoughness;',
                'float clearcoatRoughness = mix(clearcoatRoughness, max(clearcoatRoughness, 0.55), topMask);'
            )
            .replace(
                'float iridescence = iridescence;',
                'float iridescence = mix(iridescence, iridescence * 0.2, topMask);'
            )
            .replace(
                '#include <color_fragment>',
                `#include <color_fragment>
                vec3 dims = max(uDimensions, vec3(0.0001));
                float yNorm = clamp((vLocalPos.y / dims.y) * 0.5 + 0.5 + uGradientBias, 0.0, 1.0);
                float gradMix = pow(yNorm, max(uGradientSharpness, 0.001));
                vec3 gradColor = mix(uSecondaryColor, uAccentColor, gradMix);

                float stripePhase = vLocalPos.z * uStripeFrequency;
                float stripeDeriv = fwidth(stripePhase);
                float stripeFade = 1.0 - clamp(stripeDeriv * 2.5, 0.0, 1.0);
                float stripes = abs(sin(stripePhase));
                stripes = pow(stripes, 4.0) * (1.0 - abs(vWorldNormal.y));
                vec3 stripeColor = uEdgeColor * stripes * uStripeStrength * stripeFade;

                vec3 nrm = normalize(vWorldNormal);
                float topSuppression = 1.0 - topMask;
                float rim = pow(1.0 - abs(dot(nrm, vec3(0.0, 0.0, 1.0))), 3.0);
                rim += pow(1.0 - abs(dot(nrm, vec3(0.0, 1.0, 0.0))), 4.0) * 0.5;

                vec3 fresnelDir = normalize(vViewPosition);
                float fresnel = pow(1.0 - max(dot(nrm, fresnelDir), 0.0), 3.0);

                float scanPhase = (vLocalPos.y + dims.y * 0.5) * uScanlineFrequency;
                float scanline = 0.5 + 0.5 * sin(scanPhase);
                float scanFade = 1.0 - clamp(fwidth(scanPhase) * 2.0, 0.0, 1.0);
                vec3 scanColor = uAccentColor * (pow(scanline, 6.0) * uScanlineStrength * topSuppression * scanFade);

                float depthNorm = clamp((vLocalPos.z / dims.z) * 0.5 + 0.5, 0.0, 1.0);
                float glow = exp(-abs(vLocalPos.x) / max(dims.x, 0.0001) * uGlowFalloff);
                vec3 depthAccent = mix(uSecondaryColor, uAccentColor, depthNorm) * glow * uDepthAccentStrength;

                vec3 rimColor = uEdgeColor * (rim * uRimIntensity + fresnel * uFresnelBoost);

                vec3 highlightDir = normalize(vec3(0.35, 0.85, 0.2));
                float glint = pow(max(dot(nrm, highlightDir), 0.0), 16.0) * uGlintStrength * topSuppression;
                vec3 glintColor = uEdgeColor * glint;

                float holoNoise = fract(sin(dot(vLocalPos.xz, vec2(12.9898, 78.233))) * 43758.5453);
                vec3 noiseColor = uEdgeColor * pow(holoNoise, 4.0) * uNoiseStrength * topSuppression;

                diffuseColor.rgb = mix(diffuseColor.rgb, gradColor, clamp(uAccentMix, 0.0, 1.0));
                diffuseColor.rgb += stripeColor * topSuppression;
                diffuseColor.rgb += rimColor;
                diffuseColor.rgb += scanColor;
                diffuseColor.rgb += depthAccent;
                diffuseColor.rgb += glintColor;
                diffuseColor.rgb += noiseColor;
            `
            );
    };

    return material;
}

/**
 * Update the colour palette of a sci-fi material so that gradients, rim glows
 * and emissive accents stay in sync with a new target colour.
 */
export function updateSciFiMaterialColor(material, color) {
    if (!material || !color) return;
    const materials = Array.isArray(material) ? material : [material];
    for (const mat of materials) {
        if (!mat) continue;
        if (!mat.userData || !mat.userData.sciFiUniforms) {
            if (mat.color) {
                mat.color.set(color);
            }
            if (mat.emissive) {
                mat.emissive.set(color);
            }
            continue;
        }
        const uniforms = mat.userData.sciFiUniforms;
        const accent = toColor(color, color);
        accent.offsetHSL(0.02, 0.08, 0.05);
        const secondary = accent.clone();
        secondary.offsetHSL(0, -0.22, -0.48);
        const edge = accent.clone();
        edge.offsetHSL(-0.04, -0.08, 0.32);
        const emissive = accent.clone();
        emissive.offsetHSL(-0.02, -0.12, 0.28);

        uniforms.uAccentColor.value.copy(accent);
        uniforms.uSecondaryColor.value.copy(secondary);
        uniforms.uEdgeColor.value.copy(edge);

        if (mat.color) mat.color.copy(secondary);
        if (mat.emissive) mat.emissive.copy(emissive);
        if (typeof mat.emissiveIntensity === 'number') {
            mat.emissiveIntensity = Math.max(mat.emissiveIntensity, 0.48);
        }
    }
}

/**
 * Update cached dimensions used by the gradient and stripe helpers. Call this
 * when the geometry is rebuilt with different measurements.
 */
export function updateSciFiDimensions(material, dimensions) {
    if (!material || !dimensions) return;
    const materials = Array.isArray(material) ? material : [material];
    const dims = resolveDimensions(dimensions);
    for (const mat of materials) {
        if (!mat || !mat.userData || !mat.userData.sciFiUniforms) continue;
        const target = mat.userData.sciFiUniforms.uDimensions;
        if (target && target.value) {
            target.value.copy(dims);
        }
    }
}

/**
 * Update numeric sci-fi shader uniforms for a material.
 * Accepts keys like `stripeStrength` or `uStripeStrength`.
 */
export function updateSciFiMaterialUniforms(material, updates = {}) {
    if (!material || !updates) return;
    const materials = Array.isArray(material) ? material : [material];
    for (const mat of materials) {
        const uniforms = mat && mat.userData ? mat.userData.sciFiUniforms : null;
        if (!uniforms) continue;
        for (const [key, value] of Object.entries(updates)) {
            if (value === undefined || value === null) continue;
            const uniformKey = key.startsWith('u')
                ? key
                : `u${key.charAt(0).toUpperCase()}${key.slice(1)}`;
            const uniform = uniforms[uniformKey];
            if (!uniform || typeof uniform.value !== 'number') continue;
            if (typeof value === 'number' && Number.isFinite(value)) {
                uniform.value = value;
            }
        }
    }
}
