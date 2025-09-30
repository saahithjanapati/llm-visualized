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
        baseColor = 0x071226,
        accentColor = 0x49f3ff,
        secondaryColor = 0x0a1b2d,
        edgeColor = 0x9cf6ff,
        emissiveColor = 0x4ff6ff,
        emissiveIntensity = 0.45,
        metalness = 0.78,
        roughness = 0.12,
        clearcoat = 0.92,
        clearcoatRoughness = 0.22,
        transmission = 0.12,
        thickness = 1.6,
        iridescence = 0.42,
        iridescenceIOR = 1.32,
        sheen = 0.52,
        sheenColor = 0xa9ffff,
        sheenRoughness = 0.48,
        envMapIntensity = 1.65,
        transparent = true,
        opacity = 0.9,
        side = THREE.FrontSide,
        dimensions = new THREE.Vector3(1, 1, 1),
        stripeFrequency = 8.0,
        stripeStrength = 0.58,
        rimIntensity = 0.72,
        gradientSharpness = 1.55,
        gradientBias = 0.05,
        fresnelBoost = 0.4,
        circuitColor = 0x73e7ff,
        circuitFrequency = 16.0,
        circuitStrength = 0.22,
        circuitContrast = 2.6,
        circuitMix = 0.35,
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

    const dims = resolveDimensions(dimensions);

    const uniforms = {
        uAccentColor: { value: toColor(accentColor, 0x49f3ff) },
        uSecondaryColor: { value: toColor(secondaryColor, 0x0a1b2d) },
        uEdgeColor: { value: toColor(edgeColor, 0x9cf6ff) },
        uDimensions: { value: dims },
        uStripeFrequency: { value: stripeFrequency },
        uStripeStrength: { value: stripeStrength },
        uRimIntensity: { value: rimIntensity },
        uGradientSharpness: { value: gradientSharpness },
        uGradientBias: { value: gradientBias },
        uFresnelBoost: { value: fresnelBoost },
        uCircuitColor: { value: toColor(circuitColor, 0x73e7ff) },
        uCircuitFrequency: { value: circuitFrequency },
        uCircuitStrength: { value: circuitStrength },
        uCircuitContrast: { value: circuitContrast },
        uCircuitMix: { value: circuitMix }
    };

    for (const [key, value] of Object.entries(extraUniforms)) {
        uniforms[key] = value;
    }

    material.userData.sciFiUniforms = uniforms;

    material.onBeforeCompile = (shader) => {
        shader.uniforms = { ...shader.uniforms, ...uniforms };

        shader.vertexShader = shader.vertexShader
            .replace('#include <common>', '#include <common>\nvarying vec3 vLocalPos;\nvarying vec3 vWorldNormal;\n')
            .replace('#include <begin_vertex>', '#include <begin_vertex>\n    vLocalPos = position;\n    vWorldNormal = normalize(normalMatrix * normal);\n');

        shader.fragmentShader = shader.fragmentShader
            .replace(
                '#include <common>',
                '#include <common>\nvarying vec3 vLocalPos;\nvarying vec3 vWorldNormal;\nuniform vec3 uAccentColor;\nuniform vec3 uSecondaryColor;\nuniform vec3 uEdgeColor;\nuniform vec3 uDimensions;\nuniform float uStripeFrequency;\nuniform float uStripeStrength;\nuniform float uRimIntensity;\nuniform float uGradientSharpness;\nuniform float uGradientBias;\nuniform float uFresnelBoost;\nuniform vec3 uCircuitColor;\nuniform float uCircuitFrequency;\nuniform float uCircuitStrength;\nuniform float uCircuitContrast;\nuniform float uCircuitMix;\n'
            )
            .replace(
                '#include <color_fragment>',
                `#include <color_fragment>
                vec3 dims = max(uDimensions, vec3(0.0001));
                float yNorm = clamp((vLocalPos.y / dims.y) * 0.5 + 0.5 + uGradientBias, 0.0, 1.0);
                float gradMix = pow(yNorm, max(uGradientSharpness, 0.001));
                vec3 gradColor = mix(uSecondaryColor, uAccentColor, gradMix);

                float stripes = abs(sin(vLocalPos.z * uStripeFrequency));
                stripes = pow(stripes, 4.0) * (1.0 - abs(vWorldNormal.y));
                vec3 stripeColor = uEdgeColor * stripes * uStripeStrength;

                vec3 nrm = normalize(vWorldNormal);
                float rim = pow(1.0 - abs(dot(nrm, vec3(0.0, 0.0, 1.0))), 3.0);
                rim += pow(1.0 - abs(dot(nrm, vec3(0.0, 1.0, 0.0))), 4.0) * 0.5;

                vec3 fresnelDir = normalize(vViewPosition);
                float fresnel = pow(1.0 - max(dot(nrm, fresnelDir), 0.0), 3.0);

                vec3 rimColor = uEdgeColor * (rim * uRimIntensity + fresnel * uFresnelBoost);

                vec2 circuitUv = vec2(vLocalPos.x / max(dims.x, 0.0001), vLocalPos.z / max(dims.z, 0.0001));
                float circuitA = abs(sin((circuitUv.x + circuitUv.y) * uCircuitFrequency));
                float circuitB = abs(sin((circuitUv.x - circuitUv.y) * (uCircuitFrequency * 0.62)));
                float circuitC = abs(sin((vLocalPos.y / max(dims.y, 0.0001)) * uCircuitFrequency * 0.5));
                float circuitry = pow(clamp(circuitA + circuitB, 0.0, 1.0), max(uCircuitContrast, 0.001));
                circuitry = mix(circuitry, pow(clamp(circuitC, 0.0, 1.0), max(uCircuitContrast * 1.25, 0.001)), clamp(uCircuitMix, 0.0, 1.0));
                vec3 circuitColor = uCircuitColor * circuitry * uCircuitStrength;

                vec3 local01 = clamp((vLocalPos / dims) * 0.5 + 0.5, 0.0, 1.0);
                float edgeGlow = pow(1.0 - min(min(local01.x, local01.z), local01.y), 4.0);
                vec3 panelGlow = uCircuitColor * edgeGlow * (uCircuitStrength * 0.18);

                diffuseColor.rgb = mix(diffuseColor.rgb, gradColor, 0.85);
                diffuseColor.rgb += stripeColor;
                diffuseColor.rgb += rimColor;
                diffuseColor.rgb += circuitColor;
                diffuseColor.rgb += panelGlow;
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
        const secondary = accent.clone();
        secondary.offsetHSL(0, -0.18, -0.4);
        const edge = accent.clone();
        edge.offsetHSL(0, -0.05, 0.2);

        uniforms.uAccentColor.value.copy(accent);
        uniforms.uSecondaryColor.value.copy(secondary);
        uniforms.uEdgeColor.value.copy(edge);
        if (uniforms.uCircuitColor) {
            uniforms.uCircuitColor.value.copy(edge);
        }

        if (mat.color) mat.color.copy(secondary);
        if (mat.emissive) mat.emissive.copy(edge);
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
