import { updateSciFiMaterialUniforms } from './sciFiMaterial.js';

export function withMaterialArray(material, callback) {
    if (!material || typeof callback !== 'function') return;
    const mats = Array.isArray(material) ? material : [material];
    mats.forEach((mat) => {
        if (!mat) return;
        callback(mat);
    });
}

export function forEachMatrixMaterial(matrix, callback) {
    if (!matrix || typeof callback !== 'function') return;
    withMaterialArray(matrix.mesh?.material, callback);
    withMaterialArray(matrix.frontCapMesh?.material, callback);
    withMaterialArray(matrix.backCapMesh?.material, callback);
}

export function applyMatrixUserData(matrix, data) {
    if (!matrix || !data || typeof data !== 'object') return;
    const targets = [matrix.group, matrix.mesh, matrix.frontCapMesh, matrix.backCapMesh];
    targets.forEach((target) => {
        if (!target) return;
        target.userData = target.userData || {};
        Object.assign(target.userData, data);
    });
}

export function applyMatrixLabel(matrix, label) {
    if (!matrix || typeof label !== 'string') return;
    applyMatrixUserData(matrix, { label });
}

export function applyMatrixMaterialTweaks(matrix, tweaks = null, { uniforms = null } = {}) {
    if (!matrix) return;
    const {
        roughnessMin,
        metalnessMax,
        clearcoatMax,
        clearcoatRoughnessMin,
        iridescenceMax,
        envMapIntensityMax
    } = tweaks || {};

    forEachMatrixMaterial(matrix, (mat) => {
        if (!mat) return;
        if (typeof roughnessMin === 'number' && typeof mat.roughness === 'number') {
            mat.roughness = Math.max(mat.roughness, roughnessMin);
        }
        if (typeof metalnessMax === 'number' && typeof mat.metalness === 'number') {
            mat.metalness = Math.min(mat.metalness, metalnessMax);
        }
        if (typeof clearcoatMax === 'number' && typeof mat.clearcoat === 'number') {
            mat.clearcoat = Math.min(mat.clearcoat, clearcoatMax);
        }
        if (typeof clearcoatRoughnessMin === 'number' && typeof mat.clearcoatRoughness === 'number') {
            mat.clearcoatRoughness = Math.max(mat.clearcoatRoughness, clearcoatRoughnessMin);
        }
        if (typeof iridescenceMax === 'number' && typeof mat.iridescence === 'number') {
            mat.iridescence = Math.min(mat.iridescence, iridescenceMax);
        }
        if (typeof envMapIntensityMax === 'number' && typeof mat.envMapIntensity === 'number') {
            mat.envMapIntensity = Math.min(mat.envMapIntensity, envMapIntensityMax);
        }
    });

    if (!uniforms) return;
    updateSciFiMaterialUniforms(matrix.mesh?.material, uniforms);
    updateSciFiMaterialUniforms(matrix.frontCapMesh?.material, uniforms);
    updateSciFiMaterialUniforms(matrix.backCapMesh?.material, uniforms);
}
