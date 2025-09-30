import * as THREE from 'three';

const TOWER_MATERIAL_PRESETS = {
    matrixSide: {
        baseColor: 0x030812,
        accentColor: 0x31f4ff,
        secondaryColor: 0x091a2f,
        edgeColor: 0x8ef9ff,
        emissiveColor: 0x40f6ff,
        emissiveIntensity: 0.6,
        metalness: 0.82,
        roughness: 0.14,
        clearcoat: 0.92,
        clearcoatRoughness: 0.2,
        transmission: 0.2,
        thickness: 1.6,
        iridescence: 0.45,
        iridescenceIOR: 1.32,
        sheen: 0.55,
        sheenColor: 0x9ff9ff,
        sheenRoughness: 0.42,
        envMapIntensity: 1.75,
        transparent: true,
        opacity: 0.94,
        stripeStrength: 0.6,
        rimIntensity: 0.78,
        gradientSharpness: 1.55,
        gradientBias: 0.06,
        fresnelBoost: 0.48
    },
    matrixCap: {
        baseColor: 0x040a16,
        accentColor: 0x4af6ff,
        secondaryColor: 0x0b1d33,
        edgeColor: 0xa6ffff,
        emissiveColor: 0x5ef5ff,
        emissiveIntensity: 0.7,
        metalness: 0.78,
        roughness: 0.1,
        clearcoat: 0.95,
        clearcoatRoughness: 0.16,
        transmission: 0.26,
        thickness: 1.8,
        iridescence: 0.5,
        iridescenceIOR: 1.36,
        sheen: 0.6,
        sheenColor: 0xbaf8ff,
        sheenRoughness: 0.38,
        envMapIntensity: 1.9,
        transparent: true,
        opacity: 0.9,
        stripeStrength: 0.48,
        rimIntensity: 0.88,
        gradientSharpness: 1.6,
        gradientBias: 0.08,
        fresnelBoost: 0.52
    },
    matrixSlice: {
        baseColor: 0x030913,
        accentColor: 0x37f0ff,
        secondaryColor: 0x0a1c30,
        edgeColor: 0x94fbff,
        emissiveColor: 0x49f2ff,
        emissiveIntensity: 0.62,
        metalness: 0.8,
        roughness: 0.13,
        clearcoat: 0.93,
        clearcoatRoughness: 0.18,
        transmission: 0.22,
        thickness: 1.65,
        iridescence: 0.47,
        iridescenceIOR: 1.33,
        sheen: 0.57,
        sheenColor: 0xa6f7ff,
        sheenRoughness: 0.4,
        envMapIntensity: 1.8,
        transparent: true,
        opacity: 0.93,
        stripeStrength: 0.55,
        rimIntensity: 0.82,
        gradientSharpness: 1.58,
        gradientBias: 0.065,
        fresnelBoost: 0.5
    },
    matrixSliceCap: {
        baseColor: 0x040a16,
        accentColor: 0x49f4ff,
        secondaryColor: 0x0b1f34,
        edgeColor: 0xa0fdff,
        emissiveColor: 0x58f3ff,
        emissiveIntensity: 0.72,
        metalness: 0.78,
        roughness: 0.11,
        clearcoat: 0.95,
        clearcoatRoughness: 0.15,
        transmission: 0.25,
        thickness: 1.85,
        iridescence: 0.5,
        iridescenceIOR: 1.35,
        sheen: 0.62,
        sheenColor: 0xb6f8ff,
        sheenRoughness: 0.36,
        envMapIntensity: 1.92,
        transparent: true,
        opacity: 0.9,
        stripeStrength: 0.45,
        rimIntensity: 0.92,
        gradientSharpness: 1.62,
        gradientBias: 0.085,
        fresnelBoost: 0.54
    },
    layerNorm: {
        baseColor: 0x0a0716,
        accentColor: 0x8859ff,
        secondaryColor: 0x140c2e,
        edgeColor: 0xf0d4ff,
        emissiveColor: 0x7e64ff,
        emissiveIntensity: 0.62,
        metalness: 0.74,
        roughness: 0.19,
        clearcoat: 0.9,
        clearcoatRoughness: 0.24,
        transmission: 0.18,
        thickness: 1.5,
        iridescence: 0.62,
        iridescenceIOR: 1.38,
        sheen: 0.58,
        sheenColor: 0xd8c3ff,
        sheenRoughness: 0.46,
        envMapIntensity: 1.6,
        transparent: true,
        opacity: 0.9,
        stripeStrength: 0.42,
        rimIntensity: 0.88,
        gradientSharpness: 1.52,
        gradientBias: 0.055,
        fresnelBoost: 0.46
    },
    layerNormSlice: {
        baseColor: 0x0a0716,
        accentColor: 0x8f5eff,
        secondaryColor: 0x1a1036,
        edgeColor: 0xf4d9ff,
        emissiveColor: 0x8869ff,
        emissiveIntensity: 0.65,
        metalness: 0.76,
        roughness: 0.17,
        clearcoat: 0.92,
        clearcoatRoughness: 0.2,
        transmission: 0.2,
        thickness: 1.55,
        iridescence: 0.65,
        iridescenceIOR: 1.4,
        sheen: 0.6,
        sheenColor: 0xded0ff,
        sheenRoughness: 0.44,
        envMapIntensity: 1.7,
        transparent: true,
        opacity: 0.92,
        stripeStrength: 0.45,
        rimIntensity: 0.9,
        gradientSharpness: 1.55,
        gradientBias: 0.06,
        fresnelBoost: 0.48
    }
};

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

function cloneValue(value) {
    if (value instanceof THREE.Color) return value.clone();
    if (value instanceof THREE.Vector3) return value.clone();
    if (Array.isArray(value)) return value.slice();
    if (value && typeof value === 'object') return { ...value };
    return value;
}

export function towerMaterialPreset(name, overrides = {}) {
    const base = TOWER_MATERIAL_PRESETS[name] || {};
    const result = { ...base };

    if (base.dimensions) {
        result.dimensions = cloneValue(base.dimensions);
    }
    if (base.extraUniforms) {
        result.extraUniforms = { ...base.extraUniforms };
    }

    if (overrides.dimensions) {
        if (result.dimensions && !(overrides.dimensions instanceof THREE.Vector3) && typeof overrides.dimensions === 'object') {
            result.dimensions = { ...result.dimensions, ...overrides.dimensions };
        } else {
            result.dimensions = cloneValue(overrides.dimensions);
        }
    }

    if (overrides.extraUniforms) {
        result.extraUniforms = { ...(result.extraUniforms || {}), ...overrides.extraUniforms };
    }

    for (const [key, value] of Object.entries(overrides)) {
        if (key === 'dimensions' || key === 'extraUniforms') continue;
        result[key] = value;
    }

    return result;
}

/**
 * Create a MeshPhysicalMaterial configured with a custom shader that
 * introduces holographic gradients, animated-looking stripes and rim glows.
 * The shader runs entirely on the GPU and can be recoloured at runtime via
 * {@link updateSciFiMaterialColor}.
 */
export function createSciFiMaterial(options = {}) {
    const {
        baseColor = 0x060913,
        accentColor = 0x33e0ff,
        secondaryColor = 0x09111f,
        edgeColor = 0x7fffff,
        emissiveColor = 0x2fffff,
        emissiveIntensity = 0.35,
        metalness = 0.65,
        roughness = 0.18,
        clearcoat = 0.7,
        clearcoatRoughness = 0.3,
        transmission = 0.08,
        thickness = 1.2,
        iridescence = 0.35,
        iridescenceIOR = 1.25,
        sheen = 0.45,
        sheenColor = 0x99ffff,
        sheenRoughness = 0.6,
        envMapIntensity = 1.4,
        transparent = true,
        opacity = 0.9,
        side = THREE.FrontSide,
        dimensions = new THREE.Vector3(1, 1, 1),
        stripeFrequency = 8.0,
        stripeStrength = 0.45,
        rimIntensity = 0.65,
        gradientSharpness = 1.4,
        gradientBias = 0.04,
        fresnelBoost = 0.35,
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
        uAccentColor: { value: toColor(accentColor, 0x33e0ff) },
        uSecondaryColor: { value: toColor(secondaryColor, 0x09111f) },
        uEdgeColor: { value: toColor(edgeColor, 0x7fffff) },
        uDimensions: { value: dims },
        uStripeFrequency: { value: stripeFrequency },
        uStripeStrength: { value: stripeStrength },
        uRimIntensity: { value: rimIntensity },
        uGradientSharpness: { value: gradientSharpness },
        uGradientBias: { value: gradientBias },
        uFresnelBoost: { value: fresnelBoost }
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
                '#include <common>\nvarying vec3 vLocalPos;\nvarying vec3 vWorldNormal;\nuniform vec3 uAccentColor;\nuniform vec3 uSecondaryColor;\nuniform vec3 uEdgeColor;\nuniform vec3 uDimensions;\nuniform float uStripeFrequency;\nuniform float uStripeStrength;\nuniform float uRimIntensity;\nuniform float uGradientSharpness;\nuniform float uGradientBias;\nuniform float uFresnelBoost;\n'
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

                diffuseColor.rgb = mix(diffuseColor.rgb, gradColor, 0.85);
                diffuseColor.rgb += stripeColor;
                diffuseColor.rgb += rimColor;
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
