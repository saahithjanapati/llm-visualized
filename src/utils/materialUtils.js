import * as THREE from 'three';

export const GLOBAL_EMISSIVE_INTENSITY_SCALE = 0.9;

const GLOBAL_REFLECTIVITY_PROFILE = Object.freeze({
    envMapIntensityScale: 0.82,
    metalnessScale: 0.9,
    roughnessOffset: 0.07,
    clearcoatScale: 0.88,
    clearcoatRoughnessOffset: 0.07,
    iridescenceScale: 0.88,
    reflectivityScale: 0.9,
    emissiveIntensityScale: GLOBAL_EMISSIVE_INTENSITY_SCALE
});

const REFLECTIVITY_BASELINE_KEY = '__reflectivityBaselineV1';

function clampUnit(value) {
    return THREE.MathUtils.clamp(value, 0, 1);
}

export function scaleGlobalEmissiveIntensity(value) {
    const intensity = Number(value);
    if (!Number.isFinite(intensity)) return value;
    return Math.max(0, intensity * GLOBAL_EMISSIVE_INTENSITY_SCALE);
}

function captureBaseline(mat) {
    if (!mat) return null;
    if (!mat.userData) mat.userData = {};
    if (mat.userData[REFLECTIVITY_BASELINE_KEY]) return mat.userData[REFLECTIVITY_BASELINE_KEY];

    const baseline = {};
    const maybeCapture = (key) => {
        if (typeof mat[key] === 'number') baseline[key] = mat[key];
    };

    maybeCapture('envMapIntensity');
    maybeCapture('metalness');
    maybeCapture('roughness');
    maybeCapture('clearcoat');
    maybeCapture('clearcoatRoughness');
    maybeCapture('iridescence');
    maybeCapture('reflectivity');
    maybeCapture('emissiveIntensity');

    mat.userData[REFLECTIVITY_BASELINE_KEY] = baseline;
    return baseline;
}

function applyGlobalReflectivityProfile(mat, profile = GLOBAL_REFLECTIVITY_PROFILE) {
    if (!mat || !profile) return;
    const baseline = captureBaseline(mat);
    if (!baseline) return;

    if (typeof baseline.envMapIntensity === 'number' && typeof mat.envMapIntensity === 'number') {
        mat.envMapIntensity = Math.max(0, baseline.envMapIntensity * profile.envMapIntensityScale);
    }
    if (typeof baseline.metalness === 'number' && typeof mat.metalness === 'number') {
        mat.metalness = clampUnit(baseline.metalness * profile.metalnessScale);
    }
    if (typeof baseline.roughness === 'number' && typeof mat.roughness === 'number') {
        mat.roughness = clampUnit(baseline.roughness + profile.roughnessOffset);
    }
    if (typeof baseline.clearcoat === 'number' && typeof mat.clearcoat === 'number') {
        mat.clearcoat = clampUnit(baseline.clearcoat * profile.clearcoatScale);
    }
    if (typeof baseline.clearcoatRoughness === 'number' && typeof mat.clearcoatRoughness === 'number') {
        mat.clearcoatRoughness = clampUnit(baseline.clearcoatRoughness + profile.clearcoatRoughnessOffset);
    }
    if (typeof baseline.iridescence === 'number' && typeof mat.iridescence === 'number') {
        mat.iridescence = clampUnit(baseline.iridescence * profile.iridescenceScale);
    }
    if (typeof baseline.reflectivity === 'number' && typeof mat.reflectivity === 'number') {
        mat.reflectivity = clampUnit(baseline.reflectivity * profile.reflectivityScale);
    }
    if (typeof baseline.emissiveIntensity === 'number' && typeof mat.emissiveIntensity === 'number') {
        mat.emissiveIntensity = scaleGlobalEmissiveIntensity(baseline.emissiveIntensity);
    }
}

export function applyPhysicalMaterialsToScene(scene, enabled = true) {
    if (!scene || typeof scene.traverse !== 'function') return;

    const usePhysical = !!enabled;
    const fromCtor = usePhysical ? THREE.MeshStandardMaterial : THREE.MeshPhysicalMaterial;
    const toCtor = usePhysical ? THREE.MeshPhysicalMaterial : THREE.MeshStandardMaterial;

    scene.traverse((obj) => {
        if (!obj || !obj.isMesh) return;
        const swapMat = (mat) => {
            if (!(mat instanceof fromCtor)) return mat;
            const params = {
                color: mat.color?.clone?.(),
                metalness: mat.metalness ?? 0,
                roughness: mat.roughness ?? 1,
                transparent: mat.transparent,
                opacity: mat.opacity,
                emissive: mat.emissive?.clone?.(),
                emissiveIntensity: mat.emissiveIntensity ?? 0,
                flatShading: mat.flatShading,
                side: mat.side,
            };
            const newMat = new toCtor(params);
            mat.dispose();
            applyGlobalReflectivityProfile(newMat);
            return newMat;
        };
        if (Array.isArray(obj.material)) {
            obj.material = obj.material.map(swapMat);
        } else if (obj.material) {
            obj.material = swapMat(obj.material);
        }
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        mats.forEach((mat) => applyGlobalReflectivityProfile(mat));
    });
}

export default applyPhysicalMaterialsToScene;
