import * as THREE from 'three';
import {
    VECTOR_LENGTH_PRISM,
    PRISM_BASE_WIDTH,
    PRISM_BASE_DEPTH,
    PRISM_MAX_HEIGHT,
    PRISM_HEIGHT_SCALE_FACTOR,
    HIDE_INSTANCE_Y_OFFSET,
} from '../utils/constants.js';
import { computeCenteredPrismX, PRISM_INSTANCE_WIDTH_SCALE } from '../utils/prismLayout.js';
import { VectorVisualizationInstancedPrism } from './VectorVisualizationInstancedPrism.js';

// Mirror VectorVisualizationInstancedPrism sizing for identical visuals.
let _uniformCalculatedHeight = PRISM_MAX_HEIGHT * PRISM_HEIGHT_SCALE_FACTOR * 2.0;
_uniformCalculatedHeight = Math.max(0.01, _uniformCalculatedHeight);
const _prismWidthScale = PRISM_INSTANCE_WIDTH_SCALE;
const _prismDepthScale = 1.5;
const __halfBaseWidth = (PRISM_BASE_WIDTH * _prismWidthScale) / 2;
const TMP_SRC_LOCAL_MATRIX = new THREE.Matrix4();
const TMP_WORLD_MATRIX = new THREE.Matrix4();
const TMP_DST_MATRIX = new THREE.Matrix4();
const TMP_BATCH_INV_PARENT_MATRIX = new THREE.Matrix4();
const TMP_BATCH_HIDE_SCALE = new THREE.Vector3(0.001, 0.001, 0.001);

function getKeyColorCount(values) {
    const length = Array.isArray(values) ? values.length : 0;
    return Math.min(30, Math.max(1, length || 1));
}

function ensureArrayLike(values) {
    return Array.isArray(values) || ArrayBuffer.isView(values);
}

function cloneArrayLike(values) {
    if (!ensureArrayLike(values)) return [];
    return typeof values.slice === 'function' ? values.slice() : Array.from(values);
}

class BatchedVectorRef {
    constructor(batch, index) {
        this._batch = batch;
        this._index = index;
        this.isBatchedVectorRef = true;
        this.instanceCount = batch.prismCount;
        this.rawData = [];
        this.normalizedData = [];
        this.userData = {};
        this.currentKeyColors = [];
        this.group = new THREE.Object3D();
        this.group.userData = this.group.userData || {};
        this.group.userData.isVector = true;
        this.group.visible = false;
        this.mesh = null;
        this._lastPos = new THREE.Vector3();
        this._lastVisible = false;
        this._customMatrices = false;
        this._matricesInitialized = false;
    }

    updateDataInternal(newData, options = {}) {
        const copyData = options && options.copyData !== false;
        if (!ensureArrayLike(newData) || newData.length === 0) {
            this.rawData = [];
        } else {
            this.rawData = copyData ? cloneArrayLike(newData) : newData;
        }
        this._batch.updateVectorColors(this._index, this.rawData, options.colorOptions, options.cacheKeyData);
    }

    updateKeyColorsFromData(newData, numKeyColorsToSample = 30, colorGenerationOptions = null, cacheKeyData = null) {
        this.rawData = ensureArrayLike(newData) ? cloneArrayLike(newData) : [];
        this._batch.updateVectorColors(this._index, this.rawData, colorGenerationOptions, cacheKeyData, numKeyColorsToSample);
    }

    applyProcessedVisuals(processedData, numVisibleOutputUnits, colorOptionsForKeyColors, visualOptions = { setHiddenToBlack: true }, cacheKeyData = null) {
        const data = ensureArrayLike(processedData) ? cloneArrayLike(processedData) : [];
        this.rawData = data;
        this._batch.applyProcessedVisuals(
            this._index,
            data,
            numVisibleOutputUnits,
            colorOptionsForKeyColors,
            visualOptions,
            cacheKeyData
        );
    }

    setUniformColor(color) {
        this._batch.setVectorUniformColor(this._index, color);
    }

    copyColorsFrom(source) {
        this._batch.copyVectorColors(this._index, source);
        if (source && Array.isArray(source.currentKeyColors)) {
            this.currentKeyColors = source.currentKeyColors.map(c => c.clone());
        }
    }

    dispose() {
        // Batched vectors are owned by the batch; nothing to dispose per-ref.
    }

    getBaseWidthConstant() {
        return PRISM_BASE_WIDTH;
    }

    getWidthScale() {
        return _prismWidthScale;
    }

    getUniformHeight() {
        return _uniformCalculatedHeight;
    }

    setInstanceAppearance(index, yOffset, tempColor, newScale = null, markNeedsUpdate = true) {
        this._batch.setInstanceAppearance(this._index, index, yOffset, tempColor, newScale, markNeedsUpdate);
    }

    resetInstanceAppearance(index, markNeedsUpdate = false) {
        this._batch.resetInstanceAppearance(this._index, index, markNeedsUpdate);
    }

    markInstanceMatrixDirty() {
        this._batch.markInstanceMatrixDirty();
    }

    markLayoutDirty() {
        this._batch.markVectorLayoutDirty(this._index);
    }
}

export class BatchedPrismVectorSet {
    constructor({
        vectorCount,
        prismCount = VECTOR_LENGTH_PRISM,
        parentGroup = null,
        label = 'Batched Vector Set',
        raycastMetadataMode = 'perInstance',
    } = {}) {
        this.vectorCount = Math.max(1, Math.floor(vectorCount || 1));
        this.prismCount = Math.max(1, Math.floor(prismCount || VECTOR_LENGTH_PRISM));
        this.totalInstances = this.vectorCount * this.prismCount;
        this._raycastMetadataMode = raycastMetadataMode === 'perVector' ? 'perVector' : 'perInstance';
        this._vectorRefs = new Array(this.vectorCount);
        this._dummy = new THREE.Object3D();

        this._instanceBaseX = new Float32Array(this.prismCount);
        for (let i = 0; i < this.prismCount; i++) {
            this._instanceBaseX[i] = computeCenteredPrismX(i, this.prismCount, _prismWidthScale);
        }
        this._basePrismCenterY = _uniformCalculatedHeight / 2;

        const material = new THREE.MeshBasicMaterial({ color: new THREE.Color(0xffffff) });
        material.customProgramCacheKey = () => 'InstancedPrismGradientV1';
        material.onBeforeCompile = (shader) => {
            shader.uniforms.prismHalfWidth = { value: __halfBaseWidth };
            shader.vertexShader = shader.vertexShader.replace(
                '#include <common>',
                `#include <common>
attribute vec3 colorStart;
attribute vec3 colorEnd;
varying vec3 vColorStart;
varying vec3 vColorEnd;
varying float vGradientT;
uniform float prismHalfWidth;`
            );
            shader.vertexShader = shader.vertexShader.replace(
                '#include <begin_vertex>',
                `#include <begin_vertex>
    vColorStart = colorStart;
    vColorEnd   = colorEnd;
    vGradientT  = clamp( (position.x + prismHalfWidth) / (2.0 * prismHalfWidth), 0.0, 1.0 );`
            );
            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <common>',
                `#include <common>
varying vec3 vColorStart;
varying vec3 vColorEnd;
varying float vGradientT;`
            );
            shader.fragmentShader = shader.fragmentShader.replace(
                'vec4 diffuseColor = vec4( diffuse, opacity );',
                `vec3 grad = mix( vColorStart, vColorEnd, vGradientT );
    vec4 diffuseColor = vec4( grad, opacity );`
            );
        };

        const baseGeometry = new THREE.BoxGeometry(PRISM_BASE_WIDTH, 1, PRISM_BASE_DEPTH);
        this.mesh = new THREE.InstancedMesh(baseGeometry, material, this.totalInstances);
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this.mesh.userData.isVector = true;
        this.mesh.userData.label = label;
        this.mesh.userData.instanceKind = 'batchedVector';
        this.mesh.userData.prismCount = this.prismCount;
        this.mesh.userData.vectorCount = this.vectorCount;
        this.mesh.userData.raycastMetadataMode = this._raycastMetadataMode;
        if (this._raycastMetadataMode === 'perVector') {
            this.mesh.userData.vectorEntries = new Array(this.vectorCount);
            this.mesh.userData.vectorLabels = new Array(this.vectorCount).fill(label);
        } else {
            this.mesh.userData.instanceLabels = new Array(this.totalInstances).fill(label);
            this.mesh.userData.instanceEntries = new Array(this.totalInstances);
        }
        // Instances can span far from the origin; disable frustum culling to avoid popping.
        this.mesh.frustumCulled = false;
        this.mesh.visible = false;
        this._boundsDirty = false;
        this._dirtyVectorIndices = new Set();

        const colorStartArr = new Float32Array(this.totalInstances * 3);
        const colorEndArr = new Float32Array(this.totalInstances * 3);
        this.mesh.geometry.setAttribute('colorStart', new THREE.InstancedBufferAttribute(colorStartArr, 3));
        this.mesh.geometry.setAttribute('colorEnd', new THREE.InstancedBufferAttribute(colorEndArr, 3));
        this.mesh.instanceColor = new THREE.InstancedBufferAttribute(
            new Float32Array(this.totalInstances * 3),
            3
        );

        if (parentGroup) {
            parentGroup.add(this.mesh);
        }

        // Scratch vector to reuse exact color logic from VectorVisualizationInstancedPrism.
        // Seed with deterministic zero data so constructor setup never goes through
        // random/no-data fallback paths.
        const scratchSeed = new Array(this.prismCount).fill(0);
        this._scratch = new VectorVisualizationInstancedPrism(scratchSeed, new THREE.Vector3(), 30, this.prismCount);
        if (this._scratch.group) {
            this._scratch.group.visible = false;
        }
        if (this._scratch.mesh) {
            this._scratch.mesh.visible = false;
        }

        // Initialize matrices hidden
        for (let v = 0; v < this.vectorCount; v++) {
            this._writeVectorMatrices(v, new THREE.Vector3(0, 0, 0), false);
        }
        this.mesh.instanceMatrix.needsUpdate = true;
    }

    markInstanceMatrixDirty() {
        this.mesh.instanceMatrix.needsUpdate = true;
    }

    markVectorLayoutDirty(indexOrRef) {
        const idx = typeof indexOrRef === 'number'
            ? Math.floor(indexOrRef)
            : Number.isFinite(indexOrRef?._index)
                ? Math.floor(indexOrRef._index)
                : null;
        if (!Number.isFinite(idx) || idx < 0 || idx >= this.vectorCount) return false;
        this._dirtyVectorIndices.add(idx);
        return true;
    }

    _showMeshIfNeeded() {
        if (!this.mesh.visible) {
            this.mesh.visible = true;
        }
    }

    _syncMeshVisibility() {
        this.mesh.visible = this._vectorRefs.some((ref) => !!ref?.group?.visible);
    }

    getVectorRef(index) {
        const idx = Math.max(0, Math.min(this.vectorCount - 1, Math.floor(index)));
        let ref = this._vectorRefs[idx];
        if (!ref) {
            ref = new BatchedVectorRef(this, idx);
            this._vectorRefs[idx] = ref;
        }
        return ref;
    }

    updateVectorColors(index, data, colorGenerationOptions = null, cacheKeyData = null, numKeyColorsToSample = null) {
        const ref = this.getVectorRef(index);
        const wasVisible = !!ref._lastVisible;
        ref.rawData = ensureArrayLike(data) ? cloneArrayLike(data) : [];
        const numKeys = Number.isFinite(numKeyColorsToSample)
            ? Math.max(0, Math.floor(numKeyColorsToSample))
            : getKeyColorCount(ref.rawData);

        this._scratch.updateKeyColorsFromData(ref.rawData, numKeys, colorGenerationOptions, cacheKeyData || data);
        this._copyScratchColorsToVector(index);
        if (Array.isArray(this._scratch.currentKeyColors)) {
            ref.currentKeyColors = this._scratch.currentKeyColors.map(c => c.clone());
        }
        ref._customMatrices = false;
        this._writeVectorMatrices(index, ref.group.position, ref.group.visible);
        // Spawns/reuses can reposition the vector before the next batch-wide
        // sync pass; flag the instance buffer immediately so the visible copy
        // stays at the live trail frontier instead of one frame behind.
        this.mesh.instanceMatrix.needsUpdate = true;
        ref._lastPos.copy(ref.group.position);
        ref._lastVisible = ref.group.visible;
        ref._matricesInitialized = true;
        this._dirtyVectorIndices.delete(index);
        this.updateVectorRaycastInfo(index, ref);
        if (ref.group.visible && !wasVisible) {
            this._showMeshIfNeeded();
            this._invalidateBounds();
        }
    }

    applyProcessedVisuals(index, processedData, numVisibleOutputUnits, colorOptionsForKeyColors, visualOptions = { setHiddenToBlack: true }, cacheKeyData = null) {
        const ref = this.getVectorRef(index);
        const wasVisible = !!ref._lastVisible;
        ref.rawData = ensureArrayLike(processedData) ? cloneArrayLike(processedData) : [];

        // Reuse VectorVisualizationInstancedPrism logic for exact visuals
        this._scratch.applyProcessedVisuals(
            ref.rawData,
            numVisibleOutputUnits,
            colorOptionsForKeyColors,
            visualOptions,
            cacheKeyData || processedData
        );
        this._copyScratchToVector(index, ref.group.position);
        if (Array.isArray(this._scratch.currentKeyColors)) {
            ref.currentKeyColors = this._scratch.currentKeyColors.map(c => c.clone());
        }
        ref._customMatrices = true;
        ref._lastPos.copy(ref.group.position);
        ref._lastVisible = ref.group.visible;
        ref._matricesInitialized = true;
        this.updateVectorRaycastInfo(index, ref);
        if (ref.group.visible && !wasVisible) {
            this._showMeshIfNeeded();
            this._invalidateBounds();
        }
    }

    setVectorUniformColor(index, color) {
        const col = color instanceof THREE.Color ? color : new THREE.Color(color);
        const offset = index * this.prismCount * 3;
        const count = this.prismCount;
        const instanceColors = this.mesh.instanceColor.array;
        const colorStartAttr = this.mesh.geometry.getAttribute('colorStart');
        const colorEndAttr = this.mesh.geometry.getAttribute('colorEnd');
        for (let i = 0; i < count; i++) {
            const i3 = offset + i * 3;
            instanceColors[i3] = col.r;
            instanceColors[i3 + 1] = col.g;
            instanceColors[i3 + 2] = col.b;
            if (colorStartAttr && colorStartAttr.array) {
                colorStartAttr.array[i3] = col.r;
                colorStartAttr.array[i3 + 1] = col.g;
                colorStartAttr.array[i3 + 2] = col.b;
            }
            if (colorEndAttr && colorEndAttr.array) {
                colorEndAttr.array[i3] = col.r;
                colorEndAttr.array[i3 + 1] = col.g;
                colorEndAttr.array[i3 + 2] = col.b;
            }
        }
        this.mesh.instanceColor.needsUpdate = true;
        if (colorStartAttr) colorStartAttr.needsUpdate = true;
        if (colorEndAttr) colorEndAttr.needsUpdate = true;
    }

    copyVectorColors(destIndex, source) {
        const dstOffset = destIndex * this.prismCount * 3;
        const count = this.prismCount * 3;

        const dstInstanceColors = this.mesh.instanceColor.array;
        const dstCS = this.mesh.geometry.getAttribute('colorStart')?.array;
        const dstCE = this.mesh.geometry.getAttribute('colorEnd')?.array;

        if (source && source.isBatchedVectorRef && source._batch) {
            const srcIndex = source._index;
            const srcBatch = source._batch;
            const srcOffset = srcIndex * srcBatch.prismCount * 3;
            dstInstanceColors.set(srcBatch.mesh.instanceColor.array.subarray(srcOffset, srcOffset + count), dstOffset);
            if (dstCS && srcBatch.mesh.geometry.getAttribute('colorStart')?.array) {
                const srcCS = srcBatch.mesh.geometry.getAttribute('colorStart').array;
                dstCS.set(srcCS.subarray(srcOffset, srcOffset + count), dstOffset);
            }
            if (dstCE && srcBatch.mesh.geometry.getAttribute('colorEnd')?.array) {
                const srcCE = srcBatch.mesh.geometry.getAttribute('colorEnd').array;
                dstCE.set(srcCE.subarray(srcOffset, srcOffset + count), dstOffset);
            }
        } else if (source && source.mesh && source.mesh.instanceColor && source.mesh.instanceColor.array) {
            const srcColors = source.mesh.instanceColor.array;
            dstInstanceColors.set(srcColors.subarray(0, count), dstOffset);
            const srcCS = source.mesh.geometry?.getAttribute?.('colorStart')?.array;
            const srcCE = source.mesh.geometry?.getAttribute?.('colorEnd')?.array;
            if (dstCS && srcCS) dstCS.set(srcCS.subarray(0, count), dstOffset);
            if (dstCE && srcCE) dstCE.set(srcCE.subarray(0, count), dstOffset);
        }

        this.mesh.instanceColor.needsUpdate = true;
        const csAttr = this.mesh.geometry.getAttribute('colorStart');
        const ceAttr = this.mesh.geometry.getAttribute('colorEnd');
        if (csAttr) csAttr.needsUpdate = true;
        if (ceAttr) ceAttr.needsUpdate = true;
    }

    _resolveSourceSlice(source) {
        if (!source) return null;
        if (source.isBatchedVectorRef && source._batch?.mesh) {
            const srcBatch = source._batch;
            const srcPrismCount = Number.isFinite(srcBatch.prismCount)
                ? Math.max(1, Math.floor(srcBatch.prismCount))
                : this.prismCount;
            const srcIndex = Number.isFinite(source._index) ? Math.max(0, Math.floor(source._index)) : 0;
            const srcStart = srcIndex * srcPrismCount;
            const requestedCount = Number.isFinite(source.instanceCount)
                ? Math.max(1, Math.floor(source.instanceCount))
                : srcPrismCount;
            return {
                mesh: srcBatch.mesh,
                start: srcStart,
                count: Math.min(this.prismCount, srcPrismCount, requestedCount)
            };
        }

        if (source.mesh?.isInstancedMesh) {
            const sourceCount = Number.isFinite(source.instanceCount)
                ? Math.max(1, Math.floor(source.instanceCount))
                : Number.isFinite(source.mesh.count)
                    ? Math.max(1, Math.floor(source.mesh.count))
                    : this.prismCount;
            return {
                mesh: source.mesh,
                start: 0,
                count: Math.min(this.prismCount, sourceCount)
            };
        }

        return null;
    }

    copyVectorStateFrom(index, source, {
        targetPosition = null,
        sourceMatrixWorld = null,
        targetParentMatrixWorldInverse = null,
        copyData = true
    } = {}) {
        const ref = this.getVectorRef(index);
        const wasVisible = !!ref._lastVisible;
        ref.group.visible = true;
        if (targetPosition instanceof THREE.Vector3) {
            ref.group.position.copy(targetPosition);
        }

        if (copyData) {
            ref.rawData = ensureArrayLike(source?.rawData) ? cloneArrayLike(source.rawData) : [];
            ref.normalizedData = [];
            ref.currentKeyColors = Array.isArray(source?.currentKeyColors)
                ? source.currentKeyColors
                    .map((color) => (color?.isColor ? color.clone() : new THREE.Color(color)))
                    .filter((color) => color?.isColor)
                : [];
        }

        const sourceSlice = this._resolveSourceSlice(source);
        if (!sourceSlice || !sourceSlice.mesh) {
            ref._customMatrices = false;
            this._writeVectorMatrices(index, ref.group.position, ref.group.visible);
            this.mesh.instanceMatrix.needsUpdate = true;
            ref._lastPos.copy(ref.group.position);
            ref._lastVisible = ref.group.visible;
            ref._matricesInitialized = true;
            this._dirtyVectorIndices.delete(index);
            this.updateVectorRaycastInfo(index, ref);
            if (ref.group.visible && !wasVisible) {
                this._showMeshIfNeeded();
                this._invalidateBounds();
            }
            return ref;
        }

        const srcMesh = sourceSlice.mesh;
        const srcStart = Math.max(0, Math.floor(sourceSlice.start || 0));
        const srcCount = Math.max(0, Math.floor(sourceSlice.count || 0));
        const copyCount = Math.min(this.prismCount, srcCount);
        if (copyCount <= 0) {
            ref._customMatrices = false;
            this._writeVectorMatrices(index, ref.group.position, ref.group.visible);
            this.mesh.instanceMatrix.needsUpdate = true;
            ref._lastPos.copy(ref.group.position);
            ref._lastVisible = ref.group.visible;
            ref._matricesInitialized = true;
            this._dirtyVectorIndices.delete(index);
            this.updateVectorRaycastInfo(index, ref);
            if (ref.group.visible && !wasVisible) {
                this._showMeshIfNeeded();
                this._invalidateBounds();
            }
            return ref;
        }

        if (typeof srcMesh.updateMatrixWorld === 'function') srcMesh.updateMatrixWorld(true);
        if (this.mesh?.parent?.updateMatrixWorld) this.mesh.parent.updateMatrixWorld(true);

        const srcWorld = (sourceMatrixWorld && sourceMatrixWorld.isMatrix4)
            ? sourceMatrixWorld
            : srcMesh.matrixWorld;
        let dstInv = null;
        if (targetParentMatrixWorldInverse && targetParentMatrixWorldInverse.isMatrix4) {
            dstInv = targetParentMatrixWorldInverse;
        } else if (this.mesh?.parent?.matrixWorld) {
            TMP_BATCH_INV_PARENT_MATRIX.copy(this.mesh.parent.matrixWorld).invert();
            dstInv = TMP_BATCH_INV_PARENT_MATRIX;
        }

        const dstBase = index * this.prismCount;
        const dstMatrixArray = this.mesh.instanceMatrix.array;
        for (let i = 0; i < copyCount; i++) {
            srcMesh.getMatrixAt(srcStart + i, TMP_SRC_LOCAL_MATRIX);
            TMP_WORLD_MATRIX.multiplyMatrices(srcWorld, TMP_SRC_LOCAL_MATRIX);
            if (dstInv) {
                TMP_DST_MATRIX.multiplyMatrices(dstInv, TMP_WORLD_MATRIX);
            } else {
                TMP_DST_MATRIX.copy(TMP_WORLD_MATRIX);
            }
            TMP_DST_MATRIX.toArray(dstMatrixArray, (dstBase + i) * 16);
        }

        if (copyCount < this.prismCount) {
            const basePos = ref.group.position;
            for (let i = copyCount; i < this.prismCount; i++) {
                this._dummy.position.set(
                    this._instanceBaseX[i] + (basePos?.x || 0),
                    HIDE_INSTANCE_Y_OFFSET,
                    basePos?.z || 0
                );
                this._dummy.scale.copy(TMP_BATCH_HIDE_SCALE);
                this._dummy.updateMatrix();
                this._dummy.matrix.toArray(dstMatrixArray, (dstBase + i) * 16);
            }
        }
        this.mesh.instanceMatrix.needsUpdate = true;

        const dstColorOffset = dstBase * 3;
        const srcColorOffset = srcStart * 3;
        const colorValueCount = copyCount * 3;
        const dstColorLimit = this.prismCount * 3;

        const copyAttributeSlice = (attrName) => {
            const dstAttr = this.mesh.geometry.getAttribute(attrName);
            const srcAttr = srcMesh.geometry?.getAttribute?.(attrName);
            if (!dstAttr?.array || !srcAttr?.array) return;
            const srcAvailable = Math.max(0, srcAttr.array.length - srcColorOffset);
            const toCopy = Math.min(colorValueCount, srcAvailable, dstColorLimit);
            if (toCopy > 0) {
                dstAttr.array.set(srcAttr.array.subarray(srcColorOffset, srcColorOffset + toCopy), dstColorOffset);
            }
            if (toCopy < dstColorLimit) {
                const fillStart = dstColorOffset + toCopy;
                const fallbackBase = dstColorOffset + Math.max(0, toCopy - 3);
                const r = dstAttr.array[fallbackBase] ?? 0;
                const g = dstAttr.array[fallbackBase + 1] ?? 0;
                const b = dstAttr.array[fallbackBase + 2] ?? 0;
                for (let i = fillStart; i < dstColorOffset + dstColorLimit; i += 3) {
                    dstAttr.array[i] = r;
                    dstAttr.array[i + 1] = g;
                    dstAttr.array[i + 2] = b;
                }
            }
            dstAttr.needsUpdate = true;
        };

        if (this.mesh.instanceColor && this.mesh.instanceColor.array) {
            const srcInstColors = srcMesh.instanceColor?.array;
            const dstInstColors = this.mesh.instanceColor.array;
            if (srcInstColors) {
                const srcAvailable = Math.max(0, srcInstColors.length - srcColorOffset);
                const toCopy = Math.min(colorValueCount, srcAvailable, dstColorLimit);
                if (toCopy > 0) {
                    dstInstColors.set(srcInstColors.subarray(srcColorOffset, srcColorOffset + toCopy), dstColorOffset);
                }
                if (toCopy < dstColorLimit) {
                    const fillStart = dstColorOffset + toCopy;
                    const fallbackBase = dstColorOffset + Math.max(0, toCopy - 3);
                    const r = dstInstColors[fallbackBase] ?? 0;
                    const g = dstInstColors[fallbackBase + 1] ?? 0;
                    const b = dstInstColors[fallbackBase + 2] ?? 0;
                    for (let i = fillStart; i < dstColorOffset + dstColorLimit; i += 3) {
                        dstInstColors[i] = r;
                        dstInstColors[i + 1] = g;
                        dstInstColors[i + 2] = b;
                    }
                }
            }
            this.mesh.instanceColor.needsUpdate = true;
        }

        copyAttributeSlice('colorStart');
        copyAttributeSlice('colorEnd');

        ref._customMatrices = true;
        ref._lastPos.copy(ref.group.position);
        ref._lastVisible = ref.group.visible;
        ref._matricesInitialized = true;
        this._dirtyVectorIndices.delete(index);
        this.updateVectorRaycastInfo(index, ref);
        if (ref.group.visible && !wasVisible) {
            this._showMeshIfNeeded();
            this._invalidateBounds();
        }
        return ref;
    }

    updateVectorRaycastInfo(index, vectorRef = null) {
        const ref = vectorRef || this.getVectorRef(index);
        if (!ref) return;
        const label = (ref.group && ref.group.userData && ref.group.userData.label)
            || (ref.userData && ref.userData.activationData && ref.userData.activationData.label)
            || this.mesh.userData.label
            || null;
        const activationData = ref.userData ? ref.userData.activationData : null;
        const parentLane = ref.userData ? ref.userData.parentLane : null;
        const entry = {
            label,
            activationData,
            vectorRef: ref,
            headIndex: ref.userData ? ref.userData.headIndex : undefined,
            laneIndex: parentLane ? parentLane.laneIndex : undefined,
            tokenIndex: parentLane ? parentLane.tokenIndex : undefined,
            tokenLabel: parentLane ? parentLane.tokenLabel : undefined,
            layerIndex: parentLane && parentLane.layer ? parentLane.layer.index : (ref.group?.userData?.layerIndex),
            category: ref.userData ? ref.userData.vectorCategory : undefined,
        };
        if (this._raycastMetadataMode === 'perVector') {
            const vectorEntries = this.mesh.userData.vectorEntries || (this.mesh.userData.vectorEntries = new Array(this.vectorCount));
            const vectorLabels = this.mesh.userData.vectorLabels || (this.mesh.userData.vectorLabels = new Array(this.vectorCount));
            const vecIdx = Math.max(0, Math.min(this.vectorCount - 1, Math.floor(index)));
            vectorEntries[vecIdx] = entry;
            vectorLabels[vecIdx] = label;
            return;
        }

        const labels = this.mesh.userData.instanceLabels || (this.mesh.userData.instanceLabels = new Array(this.totalInstances));
        const entries = this.mesh.userData.instanceEntries || (this.mesh.userData.instanceEntries = new Array(this.totalInstances));
        const offset = index * this.prismCount;
        for (let i = 0; i < this.prismCount; i++) {
            const idx = offset + i;
            labels[idx] = label;
            entries[idx] = entry;
        }
    }

    dispose(options = {}) {
        const removeFromParent = options.removeFromParent !== false;
        try {
            if (removeFromParent && this.mesh && this.mesh.parent) {
                this.mesh.parent.remove(this.mesh);
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (this.mesh?.geometry && typeof this.mesh.geometry.dispose === 'function') {
                this.mesh.geometry.dispose();
            }
        } catch (_) { /* optional cleanup */ }
        try {
            const mat = this.mesh?.material;
            if (Array.isArray(mat)) {
                mat.forEach((m) => {
                    if (m && typeof m.dispose === 'function') m.dispose();
                });
            } else if (mat && typeof mat.dispose === 'function') {
                mat.dispose();
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (this._scratch && typeof this._scratch.dispose === 'function') {
                this._scratch.dispose();
            }
            if (this._scratch?.group?.parent) {
                this._scratch.group.parent.remove(this._scratch.group);
            }
        } catch (_) { /* optional cleanup */ }

        this._vectorRefs = [];
    }

    syncAll() {
        let boundsDirty = this._boundsDirty;
        let didMatrixUpdate = false;
        for (let i = 0; i < this.vectorCount; i++) {
            const ref = this._vectorRefs[i];
            if (!ref) continue;
            if (!ref.group.visible) {
                if (ref._lastVisible) {
                    this._writeVectorMatrices(i, ref.group.position, false);
                    ref._customMatrices = false;
                    ref._lastVisible = false;
                    ref._matricesInitialized = true;
                    boundsDirty = true;
                    didMatrixUpdate = true;
                }
                continue;
            }
            if (!ref._matricesInitialized) {
                this._writeVectorMatrices(i, ref.group.position, true);
                ref._lastPos.copy(ref.group.position);
                ref._lastVisible = true;
                ref._matricesInitialized = true;
                boundsDirty = true;
                didMatrixUpdate = true;
                continue;
            }
            if (!ref._lastVisible) {
                this._writeVectorMatrices(i, ref.group.position, true);
                ref._lastPos.copy(ref.group.position);
                ref._lastVisible = true;
                boundsDirty = true;
                didMatrixUpdate = true;
                continue;
            }
            const dx = ref.group.position.x - ref._lastPos.x;
            const dy = ref.group.position.y - ref._lastPos.y;
            const dz = ref.group.position.z - ref._lastPos.z;
            if (dx || dy || dz) {
                this._offsetVectorMatrices(i, dx, dy, dz);
                ref._lastPos.copy(ref.group.position);
                didMatrixUpdate = true;
            }
            ref._lastVisible = true;
        }
        if (didMatrixUpdate) {
            this.mesh.instanceMatrix.needsUpdate = true;
        }
        this._syncMeshVisibility();
        if (boundsDirty) {
            this.mesh.boundingSphere = null;
            this.mesh.boundingBox = null;
            this._boundsDirty = false;
        }
        if (this._dirtyVectorIndices.size) {
            this._dirtyVectorIndices.clear();
        }
    }

    syncDirty() {
        if (!this._dirtyVectorIndices || this._dirtyVectorIndices.size === 0) return;
        let boundsDirty = this._boundsDirty;
        let didMatrixUpdate = false;
        for (const i of this._dirtyVectorIndices) {
            const ref = this._vectorRefs[i];
            if (!ref) continue;
            if (!ref.group.visible) {
                if (ref._lastVisible) {
                    this._writeVectorMatrices(i, ref.group.position, false);
                    ref._customMatrices = false;
                    ref._lastVisible = false;
                    ref._matricesInitialized = true;
                    boundsDirty = true;
                    didMatrixUpdate = true;
                }
                continue;
            }
            if (!ref._matricesInitialized) {
                this._writeVectorMatrices(i, ref.group.position, true);
                ref._lastPos.copy(ref.group.position);
                ref._lastVisible = true;
                ref._matricesInitialized = true;
                boundsDirty = true;
                didMatrixUpdate = true;
                continue;
            }
            if (!ref._lastVisible) {
                this._writeVectorMatrices(i, ref.group.position, true);
                ref._lastPos.copy(ref.group.position);
                ref._lastVisible = true;
                boundsDirty = true;
                didMatrixUpdate = true;
                continue;
            }
            const dx = ref.group.position.x - ref._lastPos.x;
            const dy = ref.group.position.y - ref._lastPos.y;
            const dz = ref.group.position.z - ref._lastPos.z;
            if (dx || dy || dz) {
                this._offsetVectorMatrices(i, dx, dy, dz);
                ref._lastPos.copy(ref.group.position);
                didMatrixUpdate = true;
            }
            ref._lastVisible = true;
        }
        if (didMatrixUpdate) {
            this.mesh.instanceMatrix.needsUpdate = true;
        }
        this._syncMeshVisibility();
        if (boundsDirty) {
            this.mesh.boundingSphere = null;
            this.mesh.boundingBox = null;
            this._boundsDirty = false;
        }
        this._dirtyVectorIndices.clear();
    }

    _invalidateBounds() {
        this._boundsDirty = true;
    }

    setInstanceAppearance(vectorIndex, instanceIndex, yOffset, tempColor, newScale = null, markNeedsUpdate = true) {
        const idx = Math.max(0, Math.min(this.vectorCount - 1, Math.floor(vectorIndex)));
        if (instanceIndex < 0 || instanceIndex >= this.prismCount) return;
        const ref = this.getVectorRef(idx);
        const baseX = this._instanceBaseX[instanceIndex] + (ref.group.position?.x || 0);
        const baseZ = (ref.group.position?.z || 0);
        const baseY = this._basePrismCenterY + (ref.group.position?.y || 0);

        const isHidden = yOffset <= (HIDE_INSTANCE_Y_OFFSET + 1);
        const scale = this._dummy.scale;
        if (isHidden) {
            scale.set(0.001, 0.001, 0.001);
        } else if (newScale instanceof THREE.Vector3) {
            scale.copy(newScale);
        } else {
            scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
        }

        const posY = isHidden ? HIDE_INSTANCE_Y_OFFSET : baseY + (Number.isFinite(yOffset) ? yOffset : 0);
        this._dummy.position.set(baseX, posY, baseZ);
        this._dummy.updateMatrix();
        const instIndex = idx * this.prismCount + instanceIndex;
        this.mesh.setMatrixAt(instIndex, this._dummy.matrix);
        if (markNeedsUpdate) {
            this.mesh.instanceMatrix.needsUpdate = true;
        }

        if (tempColor instanceof THREE.Color) {
            if (this.mesh.instanceColor) {
                this.mesh.setColorAt(instIndex, tempColor);
                if (markNeedsUpdate) this.mesh.instanceColor.needsUpdate = true;
            }
        }
    }

    resetInstanceAppearance(vectorIndex, instanceIndex, markNeedsUpdate = false) {
        const idx = Math.max(0, Math.min(this.vectorCount - 1, Math.floor(vectorIndex)));
        if (instanceIndex < 0 || instanceIndex >= this.prismCount) return;
        const ref = this.getVectorRef(idx);
        const baseX = this._instanceBaseX[instanceIndex] + (ref.group.position?.x || 0);
        const baseY = this._basePrismCenterY + (ref.group.position?.y || 0);
        const baseZ = (ref.group.position?.z || 0);

        this._dummy.scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
        this._dummy.position.set(baseX, baseY, baseZ);
        this._dummy.updateMatrix();
        const instIndex = idx * this.prismCount + instanceIndex;
        this.mesh.setMatrixAt(instIndex, this._dummy.matrix);
        if (markNeedsUpdate) {
            this.mesh.instanceMatrix.needsUpdate = true;
        }
    }

    _copyScratchColorsToVector(index) {
        const offset = index * this.prismCount * 3;
        const count = this.prismCount * 3;
        const srcInstanceColors = this._scratch.mesh.instanceColor?.array;
        if (srcInstanceColors) {
            this.mesh.instanceColor.array.set(srcInstanceColors.subarray(0, count), offset);
        }
        const srcCS = this._scratch.mesh.geometry.getAttribute('colorStart')?.array;
        const srcCE = this._scratch.mesh.geometry.getAttribute('colorEnd')?.array;
        const dstCS = this.mesh.geometry.getAttribute('colorStart')?.array;
        const dstCE = this.mesh.geometry.getAttribute('colorEnd')?.array;
        if (srcCS && dstCS) dstCS.set(srcCS.subarray(0, count), offset);
        if (srcCE && dstCE) dstCE.set(srcCE.subarray(0, count), offset);
        this.mesh.instanceColor.needsUpdate = true;
        const csAttr = this.mesh.geometry.getAttribute('colorStart');
        const ceAttr = this.mesh.geometry.getAttribute('colorEnd');
        if (csAttr) csAttr.needsUpdate = true;
        if (ceAttr) ceAttr.needsUpdate = true;
    }

    _copyScratchToVector(index, position) {
        const pos = position || new THREE.Vector3();
        const srcMatrix = this._scratch.mesh.instanceMatrix.array;
        const dstMatrix = this.mesh.instanceMatrix.array;
        const base = index * this.prismCount;
        for (let i = 0; i < this.prismCount; i++) {
            const srcOffset = i * 16;
            const dstOffset = (base + i) * 16;
            dstMatrix[dstOffset] = srcMatrix[srcOffset];
            dstMatrix[dstOffset + 1] = srcMatrix[srcOffset + 1];
            dstMatrix[dstOffset + 2] = srcMatrix[srcOffset + 2];
            dstMatrix[dstOffset + 3] = srcMatrix[srcOffset + 3];
            dstMatrix[dstOffset + 4] = srcMatrix[srcOffset + 4];
            dstMatrix[dstOffset + 5] = srcMatrix[srcOffset + 5];
            dstMatrix[dstOffset + 6] = srcMatrix[srcOffset + 6];
            dstMatrix[dstOffset + 7] = srcMatrix[srcOffset + 7];
            dstMatrix[dstOffset + 8] = srcMatrix[srcOffset + 8];
            dstMatrix[dstOffset + 9] = srcMatrix[srcOffset + 9];
            dstMatrix[dstOffset + 10] = srcMatrix[srcOffset + 10];
            dstMatrix[dstOffset + 11] = srcMatrix[srcOffset + 11];
            dstMatrix[dstOffset + 12] = srcMatrix[srcOffset + 12] + pos.x;
            dstMatrix[dstOffset + 13] = srcMatrix[srcOffset + 13] + pos.y;
            dstMatrix[dstOffset + 14] = srcMatrix[srcOffset + 14] + pos.z;
            dstMatrix[dstOffset + 15] = srcMatrix[srcOffset + 15];
        }
        this._copyScratchColorsToVector(index);
        this.mesh.instanceMatrix.needsUpdate = true;
    }

    _offsetVectorMatrices(index, dx, dy, dz) {
        const dstMatrix = this.mesh.instanceMatrix.array;
        const base = index * this.prismCount;
        for (let i = 0; i < this.prismCount; i++) {
            const dstOffset = (base + i) * 16;
            dstMatrix[dstOffset + 12] += dx;
            dstMatrix[dstOffset + 13] += dy;
            dstMatrix[dstOffset + 14] += dz;
        }
    }

    _writeVectorMatrices(index, pos, visible = true) {
        const base = index * this.prismCount;
        const scale = this._dummy.scale;
        const position = this._dummy.position;
        const hide = !visible;
        for (let i = 0; i < this.prismCount; i++) {
            const instIndex = base + i;
            const x = this._instanceBaseX[i] + (pos?.x || 0);
            const y = hide ? HIDE_INSTANCE_Y_OFFSET : (this._basePrismCenterY + (pos?.y || 0));
            const z = (pos?.z || 0);
            if (hide) {
                scale.set(0.001, 0.001, 0.001);
            } else {
                scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
            }
            position.set(x, y, z);
            this._dummy.updateMatrix();
            this.mesh.setMatrixAt(instIndex, this._dummy.matrix);
        }
    }
}
