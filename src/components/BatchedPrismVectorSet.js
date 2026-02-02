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
}

export class BatchedPrismVectorSet {
    constructor({
        vectorCount,
        prismCount = VECTOR_LENGTH_PRISM,
        parentGroup = null,
        label = 'Batched Vector Set',
    } = {}) {
        this.vectorCount = Math.max(1, Math.floor(vectorCount || 1));
        this.prismCount = Math.max(1, Math.floor(prismCount || VECTOR_LENGTH_PRISM));
        this.totalInstances = this.vectorCount * this.prismCount;
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
        this.mesh.userData.instanceLabels = new Array(this.totalInstances).fill(label);
        this.mesh.userData.instanceEntries = new Array(this.totalInstances);
        // Instances can span far from the origin; disable frustum culling to avoid popping.
        this.mesh.frustumCulled = false;
        this._boundsDirty = false;

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

        // Scratch vector to reuse exact color logic from VectorVisualizationInstancedPrism
        this._scratch = new VectorVisualizationInstancedPrism(null, new THREE.Vector3(), 30, this.prismCount);
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
        ref._lastPos.copy(ref.group.position);
        ref._lastVisible = ref.group.visible;
        ref._matricesInitialized = true;
        this.updateVectorRaycastInfo(index, ref);
        if (ref.group.visible && !wasVisible) {
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

    updateVectorRaycastInfo(index, vectorRef = null) {
        const ref = vectorRef || this.getVectorRef(index);
        if (!ref) return;
        const labels = this.mesh.userData.instanceLabels || (this.mesh.userData.instanceLabels = new Array(this.totalInstances));
        const entries = this.mesh.userData.instanceEntries || (this.mesh.userData.instanceEntries = new Array(this.totalInstances));
        const offset = index * this.prismCount;
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
        for (let i = 0; i < this.prismCount; i++) {
            const idx = offset + i;
            labels[idx] = label;
            entries[idx] = entry;
        }
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
        if (boundsDirty) {
            this.mesh.boundingSphere = null;
            this.mesh.boundingBox = null;
            this._boundsDirty = false;
        }
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
