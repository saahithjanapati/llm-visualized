import * as THREE from 'three';
import { updateSciFiMaterialColor } from '../utils/sciFiMaterial.js';
import { HIDE_INSTANCE_Y_OFFSET } from '../utils/constants.js';
import { PREVIEW_VECTOR_BODY_INSTANCES } from './selectionPanelConstants.js';

const TMP_MATRIX = new THREE.Matrix4();
const TMP_POS = new THREE.Vector3();
const TMP_QUAT = new THREE.Quaternion();
const TMP_SCALE = new THREE.Vector3();

export function extractMaterialSnapshot(selectionInfo) {
    const root = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!root) return null;
    let material = null;
    let current = root;
    while (current && !material) {
        if (current.material) {
            material = Array.isArray(current.material) ? current.material.find(Boolean) : current.material;
        }
        current = current.parent;
    }
    if (!material) return null;
    return {
        color: material.color ? material.color.clone() : null,
        emissive: material.emissive && material.emissive.clone ? material.emissive.clone() : null,
        emissiveIntensity: material.emissiveIntensity,
        opacity: material.opacity,
        transparent: material.transparent,
        metalness: material.metalness,
        roughness: material.roughness,
        clearcoat: material.clearcoat,
        clearcoatRoughness: material.clearcoatRoughness,
        transmission: material.transmission,
        thickness: material.thickness,
        iridescence: material.iridescence,
        sheen: material.sheen,
        sheenColor: material.sheenColor && material.sheenColor.clone ? material.sheenColor.clone() : material.sheenColor,
        envMapIntensity: material.envMapIntensity
    };
}

export function applyMaterialSnapshot(object, snapshot) {
    if (!object || !snapshot) return;
    object.traverse((child) => {
        if (!child.material) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((mat) => {
            if (!mat) return;
            if (snapshot.color) {
                if (mat.userData?.sciFiUniforms) {
                    updateSciFiMaterialColor(mat, snapshot.color);
                } else if (mat.color?.copy) {
                    mat.color.copy(snapshot.color);
                }
            }
            if (snapshot.emissive && mat.emissive?.copy) {
                mat.emissive.copy(snapshot.emissive);
            }
            if (Number.isFinite(snapshot.emissiveIntensity)) mat.emissiveIntensity = snapshot.emissiveIntensity;
            if (Number.isFinite(snapshot.opacity)) mat.opacity = snapshot.opacity;
            if (typeof snapshot.transparent === 'boolean') mat.transparent = snapshot.transparent;
            if (Number.isFinite(snapshot.metalness)) mat.metalness = snapshot.metalness;
            if (Number.isFinite(snapshot.roughness)) mat.roughness = snapshot.roughness;
            if (Number.isFinite(snapshot.clearcoat)) mat.clearcoat = snapshot.clearcoat;
            if (Number.isFinite(snapshot.clearcoatRoughness)) mat.clearcoatRoughness = snapshot.clearcoatRoughness;
            if (Number.isFinite(snapshot.transmission)) mat.transmission = snapshot.transmission;
            if (Number.isFinite(snapshot.thickness)) mat.thickness = snapshot.thickness;
            if (Number.isFinite(snapshot.iridescence)) mat.iridescence = snapshot.iridescence;
            if (Number.isFinite(snapshot.sheen)) mat.sheen = snapshot.sheen;
            if (snapshot.sheenColor && mat.sheenColor && mat.sheenColor.copy) {
                mat.sheenColor.copy(snapshot.sheenColor);
            }
            if (Number.isFinite(snapshot.envMapIntensity)) mat.envMapIntensity = snapshot.envMapIntensity;
        });
    });
}

export function copyInstancedVectorSliceToPreview(previewVec, sourceMesh, sourceOffset = 0, sourceCount = null) {
    if (!previewVec?.mesh || !sourceMesh?.isInstancedMesh || typeof sourceMesh.getMatrixAt !== 'function') {
        return false;
    }
    const dstMesh = previewVec.mesh;
    const dstCount = Number.isFinite(previewVec.instanceCount)
        ? Math.max(1, Math.floor(previewVec.instanceCount))
        : 0;
    const srcTotal = Number.isFinite(sourceMesh.count)
        ? Math.max(0, Math.floor(sourceMesh.count))
        : Math.max(0, Math.floor(sourceMesh.instanceMatrix?.count || 0));
    const start = Math.max(0, Math.floor(sourceOffset || 0));
    const available = Math.max(0, srcTotal - start);
    const requested = Number.isFinite(sourceCount)
        ? Math.max(0, Math.floor(sourceCount))
        : dstCount;
    const copyCount = Math.min(dstCount, available, requested);
    if (copyCount <= 0) return false;

    for (let i = 0; i < copyCount; i += 1) {
        sourceMesh.getMatrixAt(start + i, TMP_MATRIX);
        dstMesh.setMatrixAt(i, TMP_MATRIX);
    }
    for (let i = copyCount; i < dstCount; i += 1) {
        TMP_MATRIX.makeScale(0.001, 0.001, 0.001);
        TMP_MATRIX.setPosition(0, HIDE_INSTANCE_Y_OFFSET, 0);
        dstMesh.setMatrixAt(i, TMP_MATRIX);
    }
    dstMesh.instanceMatrix.needsUpdate = true;

    if (sourceMesh.instanceColor?.array && dstMesh.instanceColor?.array) {
        const srcColors = sourceMesh.instanceColor.array;
        const dstColors = dstMesh.instanceColor.array;
        const srcStart = start * 3;
        const maxCopy = Math.min(copyCount * 3, srcColors.length - srcStart, dstColors.length);
        if (maxCopy > 0) {
            dstColors.set(srcColors.subarray(srcStart, srcStart + maxCopy), 0);
            dstMesh.instanceColor.needsUpdate = true;
        }
    }

    const copyAttr = (name) => {
        const srcAttr = sourceMesh.geometry?.getAttribute?.(name);
        const dstAttr = dstMesh.geometry?.getAttribute?.(name);
        if (!srcAttr?.array || !dstAttr?.array) return;
        const srcStart = start * 3;
        const maxCopy = Math.min(copyCount * 3, srcAttr.array.length - srcStart, dstAttr.array.length);
        if (maxCopy <= 0) return;
        dstAttr.array.set(srcAttr.array.subarray(srcStart, srcStart + maxCopy), 0);
        dstAttr.needsUpdate = true;
    };
    copyAttr('colorStart');
    copyAttr('colorEnd');
    return true;
}

export function copyInstancedVectorColorsToPreview(previewVec, sourceMesh, sourceOffset = 0, sourceCount = null) {
    if (!previewVec?.mesh || !sourceMesh?.isInstancedMesh) return false;
    const dstMesh = previewVec.mesh;
    const dstCount = Number.isFinite(previewVec.instanceCount)
        ? Math.max(1, Math.floor(previewVec.instanceCount))
        : 0;
    const srcTotal = Number.isFinite(sourceMesh.count)
        ? Math.max(0, Math.floor(sourceMesh.count))
        : Math.max(0, Math.floor(sourceMesh.instanceMatrix?.count || 0));
    const start = Math.max(0, Math.floor(sourceOffset || 0));
    const available = Math.max(0, srcTotal - start);
    const requested = Number.isFinite(sourceCount)
        ? Math.max(0, Math.floor(sourceCount))
        : dstCount;
    const copyCount = Math.min(dstCount, available, requested);
    if (copyCount <= 0) return false;

    let copied = false;
    if (sourceMesh.instanceColor?.array && dstMesh.instanceColor?.array) {
        const srcColors = sourceMesh.instanceColor.array;
        const dstColors = dstMesh.instanceColor.array;
        const srcStart = start * 3;
        const maxCopy = Math.min(copyCount * 3, srcColors.length - srcStart, dstColors.length);
        if (maxCopy > 0) {
            dstColors.set(srcColors.subarray(srcStart, srcStart + maxCopy), 0);
            dstMesh.instanceColor.needsUpdate = true;
            copied = true;
        }
    }

    const copyAttr = (name) => {
        const srcAttr = sourceMesh.geometry?.getAttribute?.(name);
        const dstAttr = dstMesh.geometry?.getAttribute?.(name);
        if (!srcAttr?.array || !dstAttr?.array) return;
        const srcStart = start * 3;
        const maxCopy = Math.min(copyCount * 3, srcAttr.array.length - srcStart, dstAttr.array.length);
        if (maxCopy <= 0) return;
        dstAttr.array.set(srcAttr.array.subarray(srcStart, srcStart + maxCopy), 0);
        dstAttr.needsUpdate = true;
        copied = true;
    };
    copyAttr('colorStart');
    copyAttr('colorEnd');
    return copied;
}

export function isInstancedVectorSliceInMotion(sourceMesh, sourceOffset = 0, sourceCount = null) {
    if (!sourceMesh?.isInstancedMesh || typeof sourceMesh.getMatrixAt !== 'function') {
        return false;
    }
    const srcTotal = Number.isFinite(sourceMesh.count)
        ? Math.max(0, Math.floor(sourceMesh.count))
        : Math.max(0, Math.floor(sourceMesh.instanceMatrix?.count || 0));
    const start = Math.max(0, Math.floor(sourceOffset || 0));
    const available = Math.max(0, srcTotal - start);
    const requested = Number.isFinite(sourceCount)
        ? Math.max(0, Math.floor(sourceCount))
        : available;
    const inspectCount = Math.min(available, requested);
    if (inspectCount <= 1) return false;

    let baselineY = null;
    let visibleCount = 0;
    let hiddenCount = 0;
    for (let i = 0; i < inspectCount; i += 1) {
        sourceMesh.getMatrixAt(start + i, TMP_MATRIX);
        TMP_MATRIX.decompose(TMP_POS, TMP_QUAT, TMP_SCALE);
        const hidden = TMP_POS.y <= HIDE_INSTANCE_Y_OFFSET * 0.5
            || TMP_SCALE.x < 0.01
            || TMP_SCALE.y < 0.01
            || TMP_SCALE.z < 0.01;
        if (hidden) {
            hiddenCount += 1;
            continue;
        }
        if (!Number.isFinite(TMP_POS.y)) continue;

        visibleCount += 1;
        if (baselineY === null) {
            baselineY = TMP_POS.y;
            continue;
        }

        // In stable vectors, all visible prisms share the same local Y.
        // Mid-addition vectors have per-prism Y offsets and should not be copied.
        if (Math.abs(TMP_POS.y - baselineY) > 0.25) {
            return true;
        }
    }

    if (hiddenCount > 0 && visibleCount > 0) {
        return true;
    }
    return false;
}

export function shouldSkipLiveVectorTransformCopy(vectorRef, vectorMesh, fallbackCount = null, options = {}) {
    if (options?.forceLiveCopy === true) return false;
    if (vectorRef?.userData?.qkvProcessed === true) return false;

    if (vectorRef?.isBatchedVectorRef && vectorRef._batch?.mesh) {
        const batch = vectorRef._batch;
        const batchPrismCount = Number.isFinite(batch.prismCount)
            ? Math.max(1, Math.floor(batch.prismCount))
            : Number.isFinite(fallbackCount)
                ? Math.max(1, Math.floor(fallbackCount))
                : PREVIEW_VECTOR_BODY_INSTANCES;
        const index = Number.isFinite(vectorRef._index) ? Math.max(0, Math.floor(vectorRef._index)) : 0;
        if (isInstancedVectorSliceInMotion(batch.mesh, index * batchPrismCount, batchPrismCount)) {
            return true;
        }
    }

    if (vectorRef?.mesh?.isInstancedMesh) {
        const srcCount = Number.isFinite(vectorRef.instanceCount)
            ? Math.max(1, Math.floor(vectorRef.instanceCount))
            : Number.isFinite(fallbackCount)
                ? Math.max(1, Math.floor(fallbackCount))
                : PREVIEW_VECTOR_BODY_INSTANCES;
        if (isInstancedVectorSliceInMotion(vectorRef.mesh, 0, srcCount)) {
            return true;
        }
    }

    if (vectorMesh?.isInstancedMesh) {
        const inspectCount = Number.isFinite(fallbackCount)
            ? Math.max(1, Math.floor(fallbackCount))
            : null;
        if (isInstancedVectorSliceInMotion(vectorMesh, 0, inspectCount)) {
            return true;
        }
    }

    return false;
}

export function tryCopyVectorAppearanceToPreview(vec, selectionInfo, vectorRef, vectorMesh, options = {}) {
    if (!vec || !vec.mesh) return false;
    if (shouldSkipLiveVectorTransformCopy(vectorRef, vectorMesh, vec.instanceCount, options)) {
        return false;
    }
    let copied = false;

    if (!copied && vectorRef?.isBatchedVectorRef && vectorRef._batch?.mesh) {
        const batch = vectorRef._batch;
        const batchPrismCount = Number.isFinite(batch.prismCount)
            ? Math.max(1, Math.floor(batch.prismCount))
            : vec.instanceCount;
        const index = Number.isFinite(vectorRef._index) ? Math.max(0, Math.floor(vectorRef._index)) : 0;
        copied = copyInstancedVectorSliceToPreview(vec, batch.mesh, index * batchPrismCount, batchPrismCount);
    }

    if (!copied && vectorRef?.mesh?.isInstancedMesh) {
        const srcCount = Number.isFinite(vectorRef.instanceCount)
            ? Math.max(1, Math.floor(vectorRef.instanceCount))
            : undefined;
        copied = copyInstancedVectorSliceToPreview(vec, vectorRef.mesh, 0, srcCount);
    }

    if (!copied && vectorMesh?.isInstancedMesh) {
        copied = copyInstancedVectorSliceToPreview(vec, vectorMesh, 0, vec.instanceCount);
    }

    if (!copied) return false;

    if (Array.isArray(vectorRef?.currentKeyColors) && vectorRef.currentKeyColors.length >= 2) {
        vec.currentKeyColors = vectorRef.currentKeyColors
            .map((color) => (color?.isColor ? color.clone() : new THREE.Color(color)))
            .filter((color) => color?.isColor);
        vec.numSubsections = Math.max(1, vec.currentKeyColors.length - 1);
    }

    const snapshot = extractMaterialSnapshot(selectionInfo);
    if (snapshot) {
        applyMaterialSnapshot(vec.group, snapshot);
    }
    return true;
}
