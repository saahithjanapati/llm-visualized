import * as THREE from 'three';

const ACTIVE_MATERIALS = new Set();
let _elapsedTime = 0;

const DEFAULT_CONFIG = {
    gradientMin: -0.5,
    gradientMax: 0.5,
    gradientColorA: '#041226',
    gradientColorB: '#27c8ff',
    gradientMix: 0.75,
    rimColor: '#8df7ff',
    rimStrength: 0.6,
    rimPower: 2.4,
    scanColor: '#58ffe6',
    scanFrequency: 5.0,
    scanSpeed: 1.4,
    scanSharpness: 4.0,
    scanIntensity: 0.8,
    gridColor: '#123b9d',
    gridDensity: 0.02,
    gridIntensity: 0.35,
    noiseScale: 3.2,
    noiseSpeed: 1.1,
    noiseIntensity: 0.18,
    noiseEmission: 0.12,
    emissiveScanBoost: 0.7,
    emissiveRimBoost: 0.45,
    baseEmissiveBoost: 0.12,
    pulseFrequency: 0.8,
    pulseAmplitude: 0.6,
    transparent: true,
    opacity: 0.92,
    emissiveColor: '#0fd4ff',
    emissiveIntensity: 0.55,
    side: THREE.FrontSide,
    metalness: 0.65,
    roughness: 0.25,
    clearcoat: 0.95,
    clearcoatRoughness: 0.08,
    envMapIntensity: 1.35,
    transmission: 0.08,
    thickness: 1.85,
    ior: 1.45,
    iridescence: 0.22,
    iridescenceIOR: 1.15,
    iridescenceThicknessRange: [80, 220],
    sheen: 1.0,
    sheenColor: '#1d8bff',
    sheenRoughness: 0.55,
    depthWrite: false
};

function buildUniforms(config) {
    return {
        uTime: { value: 0 },
        uGradientMin: { value: config.gradientMin },
        uGradientMax: { value: config.gradientMax },
        uGradientColorA: { value: new THREE.Color(config.gradientColorA) },
        uGradientColorB: { value: new THREE.Color(config.gradientColorB) },
        uGradientMix: { value: config.gradientMix },
        uRimColor: { value: new THREE.Color(config.rimColor) },
        uRimStrength: { value: config.rimStrength },
        uRimPower: { value: config.rimPower },
        uScanColor: { value: new THREE.Color(config.scanColor) },
        uScanFrequency: { value: config.scanFrequency },
        uScanSpeed: { value: config.scanSpeed },
        uScanSharpness: { value: config.scanSharpness },
        uScanIntensity: { value: config.scanIntensity },
        uGridColor: { value: new THREE.Color(config.gridColor) },
        uGridDensity: { value: config.gridDensity },
        uGridIntensity: { value: config.gridIntensity },
        uNoiseScale: { value: config.noiseScale },
        uNoiseSpeed: { value: config.noiseSpeed },
        uNoiseIntensity: { value: config.noiseIntensity },
        uNoiseEmission: { value: config.noiseEmission },
        uEmissiveScanBoost: { value: config.emissiveScanBoost },
        uEmissiveRimBoost: { value: config.emissiveRimBoost },
        uBaseEmissiveBoost: { value: config.baseEmissiveBoost },
        uPulseFrequency: { value: config.pulseFrequency },
        uPulseAmplitude: { value: config.pulseAmplitude }
    };
}

function injectSciFiShader(material, uniforms) {
    material.customProgramCacheKey = () => 'SciFiStackMaterial_v1';
    material.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, uniforms);

        if (!shader.vertexShader.includes('vSciFiLocalPosition')) {
            shader.vertexShader = shader.vertexShader.replace(
                '#include <common>',
                '#include <common>\nvarying vec3 vSciFiLocalPosition;'
            );
        }

        if (!shader.vertexShader.includes('vSciFiLocalPosition = transformed')) {
            shader.vertexShader = shader.vertexShader.replace(
                '#include <begin_vertex>',
                '#include <begin_vertex>\n\tvSciFiLocalPosition = transformed;'
            );
        }

        const uniformBlock = `uniform float uTime;\n` +
            `uniform float uGradientMin;\n` +
            `uniform float uGradientMax;\n` +
            `uniform vec3 uGradientColorA;\n` +
            `uniform vec3 uGradientColorB;\n` +
            `uniform float uGradientMix;\n` +
            `uniform vec3 uRimColor;\n` +
            `uniform float uRimStrength;\n` +
            `uniform float uRimPower;\n` +
            `uniform vec3 uScanColor;\n` +
            `uniform float uScanFrequency;\n` +
            `uniform float uScanSpeed;\n` +
            `uniform float uScanSharpness;\n` +
            `uniform float uScanIntensity;\n` +
            `uniform vec3 uGridColor;\n` +
            `uniform float uGridDensity;\n` +
            `uniform float uGridIntensity;\n` +
            `uniform float uNoiseScale;\n` +
            `uniform float uNoiseSpeed;\n` +
            `uniform float uNoiseIntensity;\n` +
            `uniform float uNoiseEmission;\n` +
            `uniform float uEmissiveScanBoost;\n` +
            `uniform float uEmissiveRimBoost;\n` +
            `uniform float uBaseEmissiveBoost;\n` +
            `uniform float uPulseFrequency;\n` +
            `uniform float uPulseAmplitude;`;

        if (!shader.fragmentShader.includes('vSciFiLocalPosition')) {
            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <common>',
                `#include <common>\nvarying vec3 vSciFiLocalPosition;\n${uniformBlock}`
            );
        }

        const sciFiChunk = `\n        float sciFiRange = max(0.0001, uGradientMax - uGradientMin);\n` +
            `        float sciFiT = clamp((vSciFiLocalPosition.y - uGradientMin) / sciFiRange, 0.0, 1.0);\n` +
            `        vec3 sciFiGradient = mix(uGradientColorA, uGradientColorB, sciFiT);\n` +
            `        diffuseColor.rgb = mix(diffuseColor.rgb, sciFiGradient, uGradientMix);\n` +
            `        float sciFiScanPhase = sciFiT * uScanFrequency * 6.28318530718 + uTime * uScanSpeed;\n` +
            `        float sciFiScan = sin(sciFiScanPhase);\n` +
            `        float sciFiBand = pow(max(0.0, 1.0 - abs(sciFiScan)), uScanSharpness) * uScanIntensity;\n` +
            `        float sciFiGrid = abs(sin((vSciFiLocalPosition.x + vSciFiLocalPosition.z) * uGridDensity));\n` +
            `        sciFiGrid = pow(max(0.0, 1.0 - sciFiGrid), 3.0) * uGridIntensity;\n` +
            `        float sciFiNoise = sin(dot(vSciFiLocalPosition, vec3(0.65, 0.45, 0.78)) * uNoiseScale + uTime * uNoiseSpeed);\n` +
            `        float sciFiNoiseMask = clamp(0.5 + 0.5 * sciFiNoise, 0.0, 1.0) * uNoiseIntensity;\n` +
            `        vec3 sciFiViewDir = normalize(vViewPosition);\n` +
            `        vec3 sciFiNormal = normalize(normal);\n` +
            `        float sciFiRim = pow(max(0.0, 1.0 - abs(dot(sciFiNormal, sciFiViewDir))), uRimPower) * uRimStrength;\n` +
            `        float sciFiPulse = 0.5 + 0.5 * sin(uTime * uPulseFrequency);\n` +
            `        diffuseColor.rgb += sciFiBand * uScanColor;\n` +
            `        diffuseColor.rgb += sciFiRim * uRimColor;\n` +
            `        diffuseColor.rgb += sciFiGrid * uGridColor;\n` +
            `        diffuseColor.rgb += sciFiNoiseMask * sciFiGradient;\n` +
            `        totalEmissiveRadiance += sciFiGradient * uBaseEmissiveBoost;\n` +
            `        totalEmissiveRadiance += sciFiBand * uScanColor * (uEmissiveScanBoost + uPulseAmplitude * sciFiPulse);\n` +
            `        totalEmissiveRadiance += sciFiRim * uRimColor * (uEmissiveRimBoost + 0.25 * uPulseAmplitude * sciFiPulse);\n` +
            `        totalEmissiveRadiance += sciFiNoiseMask * sciFiGradient * uNoiseEmission;\n`;

        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <emissivemap_fragment>',
            `#include <emissivemap_fragment>${sciFiChunk}`
        );
    };
}

function applyMaterialConfig(material, config) {
    material.transparent = config.transparent;
    material.opacity = config.opacity;
    material.side = config.side;
    material.depthWrite = config.depthWrite;
    material.color = material.color || new THREE.Color();
    if (config.baseColor) {
        material.color.set(config.baseColor);
    }
    material.metalness = config.metalness;
    material.roughness = config.roughness;
    material.clearcoat = config.clearcoat;
    material.clearcoatRoughness = config.clearcoatRoughness;
    material.envMapIntensity = config.envMapIntensity;
    material.transmission = config.transmission;
    material.thickness = config.thickness;
    material.ior = config.ior;
    material.iridescence = config.iridescence;
    material.iridescenceIOR = config.iridescenceIOR;
    if (Array.isArray(config.iridescenceThicknessRange) && config.iridescenceThicknessRange.length === 2) {
        material.iridescenceThicknessRange = [...config.iridescenceThicknessRange];
    }
    if (material.sheen !== undefined) {
        material.sheen = config.sheen;
    }
    if (material.sheenColor !== undefined) {
        material.sheenColor = material.sheenColor || new THREE.Color();
        material.sheenColor.set(config.sheenColor);
    }
    if (material.sheenRoughness !== undefined) {
        material.sheenRoughness = config.sheenRoughness;
    }
    material.emissive.set(config.emissiveColor);
    material.emissiveIntensity = config.emissiveIntensity;
}

function registerMaterial(material, config, uniforms) {
    if (!material) return material;

    if (material.userData.__sciFiHandle) {
        ACTIVE_MATERIALS.delete(material.userData.__sciFiHandle);
    }

    const handle = { material, config, uniforms };
    material.userData.__sciFiHandle = handle;
    material.userData.__sciFiConfig = { ...config };
    ACTIVE_MATERIALS.add(handle);

    const disposeHandler = () => {
        ACTIVE_MATERIALS.delete(handle);
        material.removeEventListener('dispose', disposeHandler);
    };
    material.addEventListener('dispose', disposeHandler);

    material.needsUpdate = true;
    return material;
}

export function decorateSciFiMaterial(material, options = {}) {
    if (!material) return material;
    const config = { ...DEFAULT_CONFIG, ...options };
    const uniforms = buildUniforms(config);
    applyMaterialConfig(material, config);
    injectSciFiShader(material, uniforms);
    return registerMaterial(material, config, uniforms);
}

export function createSciFiMaterial(options = {}) {
    const config = { ...DEFAULT_CONFIG, ...options };
    const material = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(options.baseColor || '#0b1b3a')
    });
    return decorateSciFiMaterial(material, config);
}

export function cloneSciFiMaterial(material, overrides = {}) {
    if (!material) return null;
    const baseConfig = material.userData?.__sciFiConfig || {};
    const merged = { ...baseConfig, ...overrides };
    const clone = material.clone();
    return decorateSciFiMaterial(clone, merged);
}

export function updateSciFiGradientRange(material, min, max) {
    const handle = material?.userData?.__sciFiHandle;
    if (!handle) return;
    handle.uniforms.uGradientMin.value = min;
    handle.uniforms.uGradientMax.value = max;
    handle.config.gradientMin = min;
    handle.config.gradientMax = max;
}

export function tickSciFiMaterials(deltaSeconds) {
    if (ACTIVE_MATERIALS.size === 0) return;
    _elapsedTime += deltaSeconds;
    for (const entry of ACTIVE_MATERIALS) {
        entry.uniforms.uTime.value = _elapsedTime;
    }
}
