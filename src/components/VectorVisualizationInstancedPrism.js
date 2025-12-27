import * as THREE from 'three';
import {
    VECTOR_LENGTH_PRISM,
    PRISM_BASE_WIDTH,
    PRISM_BASE_DEPTH,
    PRISM_MAX_HEIGHT,
    PRISM_HEIGHT_SCALE_FACTOR,
    HIDE_INSTANCE_Y_OFFSET,
    PRISM_DIMENSIONS_PER_UNIT // Added for grouping visible units
} from '../utils/constants.js';
import { computeCenteredPrismX, PRISM_INSTANCE_WIDTH_SCALE } from '../utils/prismLayout.js';
import { mapValueToColor } from '../utils/colors.js';
import { perfStats } from '../utils/perfStats.js';

// Helper for monochromatic colors
function mapValueToMonochromaticColor(value, baseHue, saturation, minLightness, maxLightness, valueMin = 0, valueMax = 1) {
    const safeMin = Number.isFinite(valueMin) ? valueMin : 0;
    const safeMax = Number.isFinite(valueMax) ? valueMax : 1;
    const fallback = safeMin + (safeMax - safeMin) * 0.5;
    const rawValue = Number.isFinite(value) ? value : fallback;
    const denom = safeMax - safeMin;
    const t = denom > 0
        ? (THREE.MathUtils.clamp(rawValue, safeMin, safeMax) - safeMin) / denom
        : 0.5;
    const lightness = THREE.MathUtils.lerp(minLightness, maxLightness, t);
    return new THREE.Color().setHSL(baseHue, saturation, lightness);
}

// Base geometry has height 1, will be scaled by data
const basePrismGeometry = new THREE.BoxGeometry(PRISM_BASE_WIDTH, 1, PRISM_BASE_DEPTH);

// Pre-calculate fixed uniform height and scales for the prisms
let _uniformCalculatedHeight = PRISM_MAX_HEIGHT * PRISM_HEIGHT_SCALE_FACTOR * 2.0; // Double height
_uniformCalculatedHeight = Math.max(0.01, _uniformCalculatedHeight); // Ensure minimum
const _prismWidthScale = PRISM_INSTANCE_WIDTH_SCALE;
const _prismDepthScale = 1.5;
// Precompute half base width used in shader patch
const __halfBaseWidth = (PRISM_BASE_WIDTH * _prismWidthScale) / 2;

const COLOR_CACHE = new WeakMap();

function isArrayLike(data) {
    return Array.isArray(data) || ArrayBuffer.isView(data);
}

function formatOptionNumber(value) {
    if (!Number.isFinite(value)) return '0';
    return value.toFixed(4);
}

function buildColorCacheKey(numKeyColors, instanceCount, colorGenerationOptions) {
    const safeNum = Math.max(0, Math.floor(numKeyColors || 0));
    const safeCount = Math.max(1, Math.floor(instanceCount || 1));
    let key = `n:${safeNum}|c:${safeCount}`;
    if (!colorGenerationOptions || typeof colorGenerationOptions !== 'object') {
        return `${key}|t:default`;
    }
    if (colorGenerationOptions.type === 'monochromatic') {
        key += `|t:mono|h:${formatOptionNumber(colorGenerationOptions.baseHue)}`;
        key += `|s:${formatOptionNumber(colorGenerationOptions.saturation)}`;
        key += `|min:${formatOptionNumber(colorGenerationOptions.minLightness)}`;
        key += `|max:${formatOptionNumber(colorGenerationOptions.maxLightness)}`;
        return key;
    }
    const type = colorGenerationOptions.type ? String(colorGenerationOptions.type) : 'custom';
    return `${key}|t:${type}`;
}

function getColorCacheEntry(data, cacheKey) {
    if (!isArrayLike(data)) return null;
    const bucket = COLOR_CACHE.get(data);
    if (!bucket) return null;
    return bucket.get(cacheKey) || null;
}

function setColorCacheEntry(data, cacheKey, entry) {
    if (!isArrayLike(data)) return;
    let bucket = COLOR_CACHE.get(data);
    if (!bucket) {
        bucket = new Map();
        COLOR_CACHE.set(data, bucket);
    }
    bucket.set(cacheKey, entry);
}

export class VectorVisualizationInstancedPrism {
    constructor(initialData = null, initialPosition = new THREE.Vector3(0, 0, 0), numSubsections = 30, instanceCount = VECTOR_LENGTH_PRISM) {
        this.group = new THREE.Group();
        this.group.position.copy(initialPosition);
        this.group.userData.label = 'Vector';

        this.instanceCount = Math.max(1, Math.floor(instanceCount));
        this.rawData = initialData || this.generateTestData();
        this.normalizedData = []; // Will be populated by updateDataInternal
        this.numSubsections = numSubsections;
        this.currentKeyColors = []; // To store the current random key colors for subsections

        // For storing individual instance animation states if needed by an external controller
        this.instanceUserData = Array(this.instanceCount).fill(null).map(() => ({})); 
        // Reuse temp objects to avoid per-frame allocations in animation updates.
        this._scratchMatrix = new THREE.Matrix4();
        this._scratchPosition = new THREE.Vector3();
        this._scratchQuaternion = new THREE.Quaternion();
        this._scratchScale = new THREE.Vector3();
        this._scratchDummy = new THREE.Object3D();

        // Create a material per vector so we can vary opacity independently,
        // but force program reuse by returning a stable cache key.
        const material = new THREE.MeshBasicMaterial({ 
            color: new THREE.Color(0xffffff)
        });
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

        const instancedGeometry = basePrismGeometry.clone();
        this.mesh = new THREE.InstancedMesh(instancedGeometry, material, this.instanceCount);
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this.group.add(this.mesh);
        
        // ------------------------------------------------------------
        // Gradient setup – each prism will smoothly blend from the
        // colour on its left edge (colorStart) to the colour on its
        // right edge (colorEnd).  We store these as per-instance
        // InstancedBufferAttributes; the shared material's shader patch
        // reads them to produce a left→right gradient.
        // ------------------------------------------------------------

        // Create (r,g,b) attribute arrays – one entry per instance
        const colorStartArr = new Float32Array(this.instanceCount * 3);
        const colorEndArr   = new Float32Array(this.instanceCount * 3);
        this.mesh.geometry.setAttribute('colorStart', new THREE.InstancedBufferAttribute(colorStartArr, 3));
        this.mesh.geometry.setAttribute('colorEnd',   new THREE.InstancedBufferAttribute(colorEndArr,   3));


        this.updateDataInternal(this.rawData.slice()); // Process initial data
        this._generateKeyColors(); // Generate initial key colors
        this.updateInstanceGeometryAndColors();      // Set initial visual state (fixed dimensions, subsection colors)
    }

    generateTestData() {
        const data = [];
        for (let i = 0; i < this.instanceCount; i++) {
            data.push(Math.random() * 2 - 1); // Values between -1 and 1 for more dynamic range
        }
        return data;
    }

    minMaxNormalize(dataArray) {
        if (!dataArray || dataArray.length === 0) return [];
        const finiteData = dataArray.filter(val => typeof val === 'number' && isFinite(val));
        if (finiteData.length === 0) return dataArray.map(() => 0.5); // Default if no finite numbers

        let minVal = finiteData[0];
        let maxVal = finiteData[0];
        for (let i = 1; i < finiteData.length; i++) {
            if (finiteData[i] < minVal) minVal = finiteData[i];
            if (finiteData[i] > maxVal) maxVal = finiteData[i];
        }
        const range = maxVal - minVal;
        if (range < 1e-7) return dataArray.map(val => (typeof val === 'number' && isFinite(val) ? 0.5 : 0.5));
        return dataArray.map(val => {
            if (typeof val === 'number' && isFinite(val)) {
                return (val - minVal) / range;
            } else {
                return 0.5; // Default for non-finite original values
            }
        });
    }

    layerNormalize(vectorData) {
        // Existing layerNormalize logic - assumed to return values in a typical range (e.g., roughly -1 to 1, or 0 to 1 after some scaling)
        // For visualization, the output of layerNormalize should ideally be consistently in 0-1 range or be minMaxNormalized before height scaling.
        // Let's assume it gives values that can be scaled. If it already produces 0-1, that's fine.
        if (!vectorData || vectorData.length === 0) return [];
        const finiteData = vectorData.filter(val => typeof val === 'number' && isFinite(val));
        if (finiteData.length === 0) return vectorData.map(() => 0.5);
        let minVal = finiteData[0];
        let maxVal = finiteData[0];
        for (let i = 1; i < finiteData.length; i++) {
            if (finiteData[i] < minVal) minVal = finiteData[i];
            if (finiteData[i] > maxVal) maxVal = finiteData[i];
        }
        const range = maxVal - minVal;
        if (range < 1e-7) return vectorData.map(val => (typeof val === 'number' && isFinite(val) ? 0.5 : 0.5));
        return vectorData.map(val => {
            if (typeof val === 'number' && isFinite(val)) {
                return (val - minVal) / range; // Ensures 0-1 output for layerNorm too
            } else {
                return 0.5;
            }
        });
    }
    
    // This method now ALWAYS applies fixed dimensions. Colors are based on subsections.
    // It's called for initial setup and when numSubsections changes.
    // Temporary visual changes (like animation offsets/colors) will be done by setInstanceAppearance.
    updateInstanceGeometryAndColors() {
        const dummy = new THREE.Object3D();

        // Prepare attribute references for gradient colours
        const colorStartAttr = this.mesh.geometry.getAttribute('colorStart');
        const colorEndAttr   = this.mesh.geometry.getAttribute('colorEnd');

        // Use the stored key colors
        if (this.currentKeyColors.length === 0) this._generateKeyColors();

        for (let i = 0; i < this.instanceCount; i++) {
            // Apply fixed, uniform dimensions
            const x = computeCenteredPrismX(i, this.instanceCount, _prismWidthScale);
            dummy.scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
            dummy.position.set(x, _uniformCalculatedHeight / 2, 0); 
            dummy.updateMatrix();
            this.mesh.setMatrixAt(i, dummy.matrix);

            // Set colour using the stored key colours and getDefaultColorForIndex logic
            const midColor = this.getDefaultColorForIndex(i);
            this.mesh.setColorAt(i, midColor);

            // Determine gradient edge colours
            const leftColor  = this.getDefaultColorForIndex(Math.max(0, i - 1));
            const rightColor = this.getDefaultColorForIndex(Math.min(this.instanceCount - 1, i + 1));

            // Store into instanced buffer attributes
            colorStartAttr.setXYZ(i, leftColor.r, leftColor.g, leftColor.b);
            colorEndAttr.setXYZ(  i, rightColor.r, rightColor.g, rightColor.b);
        }

        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.mesh.instanceColor) {
            this.mesh.instanceColor.needsUpdate = true;
        } else {
            // This warning should ideally not appear with correct material setup
            console.warn("instanceColor buffer does NOT exist on the mesh after setColorAt!");
        }

        // Mark gradient attributes "dirty" so Three.js re-uploads them
        if (colorStartAttr) colorStartAttr.needsUpdate = true;
        if (colorEndAttr)   colorEndAttr.needsUpdate   = true;
    }

    // Updates the visual appearance of a single instance for animation purposes
    setInstanceAppearance(index, yOffset, tempColor, newScale = null, markNeedsUpdate = true) {
        if (index < 0 || index >= this.instanceCount) return;
        const currentMatrix = this._scratchMatrix;
        this.mesh.getMatrixAt(index, currentMatrix);
        const position = this._scratchPosition;
        const quaternion = this._scratchQuaternion;
        const scale = this._scratchScale;
        currentMatrix.decompose(position, quaternion, scale);
        
        const baseX = computeCenteredPrismX(index, this.instanceCount, _prismWidthScale);
        const basePrismYPos = _uniformCalculatedHeight / 2; // Assuming yOffset is additive to this base for animation

        position.set(baseX, basePrismYPos + yOffset, 0);

        const isHidden = yOffset <= (HIDE_INSTANCE_Y_OFFSET + 1); // treat values at or below the sentinel as hidden

        if (isHidden) {
            // When hiding, collapse scale so nothing is rendered even if the
            // instance remains inside the camera frustum.  Position is already
            // far below the tower, but the previous implementation left the
            // default scale in place which meant the prisms could still be seen
            // when the user zoomed far out.  Shrinking them removes the
            // artefact entirely.
            scale.set(0.001, 0.001, 0.001);
            position.y = HIDE_INSTANCE_Y_OFFSET;
        } else if (newScale instanceof THREE.Vector3) {
            scale.copy(newScale);
        } else {
            // Ensure scale is default if not specified, in case it was previously altered by another call
            // This part is tricky if multiple effects are layered. For now, assume newScale is authoritative if provided.
            // If not provided, it re-uses the one from decompose, which SHOULD be the default if other methods reset it.
            // Let's explicitly set to default if no newScale, to be safe during complex animations.
            scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
        }

        currentMatrix.compose(position, quaternion, scale);
        this.mesh.setMatrixAt(index, currentMatrix);
        if (markNeedsUpdate) {
            this.mesh.instanceMatrix.needsUpdate = true;
        }

        if (tempColor instanceof THREE.Color) {
            this.mesh.setColorAt(index, tempColor);
            if (markNeedsUpdate && this.mesh.instanceColor) {
                this.mesh.instanceColor.needsUpdate = true;
            }
        }
    }

    // Resets a single instance to its default appearance (fixed dimensions, subsection color)
    resetInstanceAppearance(index, markNeedsUpdate = false) {
        if (index < 0 || index >= this.instanceCount) return;
        
        const dummy = this._scratchDummy;
        const x = computeCenteredPrismX(index, this.instanceCount, _prismWidthScale);
        dummy.scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
        dummy.position.set(x, _uniformCalculatedHeight / 2, 0); 
        dummy.updateMatrix();
        this.mesh.setMatrixAt(index, dummy.matrix);
        if (markNeedsUpdate) {
            this.mesh.instanceMatrix.needsUpdate = true;
        }
    }

    setUniformColor(color) {
        if (!this.mesh) return;
        const col = color instanceof THREE.Color ? color : new THREE.Color(color);
        const count = this.instanceCount;
        const colorStartAttr = this.mesh.geometry.getAttribute('colorStart');
        const colorEndAttr = this.mesh.geometry.getAttribute('colorEnd');

        if (!this.mesh.instanceColor) {
            this.mesh.instanceColor = new THREE.InstancedBufferAttribute(
                new Float32Array(this.instanceCount * 3),
                3
            );
        }
        const instanceColors = this.mesh.instanceColor.array;
        for (let i = 0; i < count; i++) {
            const i3 = i * 3;
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

    copyColorsFrom(source) {
        if (!source || !source.mesh || !this.mesh) return;
        const srcMesh = source.mesh;
        const dstMesh = this.mesh;
        const srcCount = Number.isFinite(source.instanceCount) ? source.instanceCount : 0;
        const dstCount = Number.isFinite(this.instanceCount) ? this.instanceCount : 0;
        const count = Math.min(srcCount, dstCount);
        if (count <= 0) return;

        const srcInstanceColor = srcMesh.instanceColor;
        if (srcInstanceColor && srcInstanceColor.array) {
            if (!dstMesh.instanceColor) {
                dstMesh.instanceColor = new THREE.InstancedBufferAttribute(
                    new Float32Array(dstCount * 3),
                    3
                );
            }
            const srcArray = srcInstanceColor.array;
            const dstArray = dstMesh.instanceColor.array;
            const copyLen = Math.min(srcArray.length, dstArray.length, count * 3);
            for (let i = 0; i < copyLen; i++) {
                dstArray[i] = srcArray[i];
            }
            if (dstArray.length > copyLen && copyLen >= 3) {
                const lastR = dstArray[copyLen - 3];
                const lastG = dstArray[copyLen - 2];
                const lastB = dstArray[copyLen - 1];
                for (let i = copyLen; i < dstArray.length; i += 3) {
                    dstArray[i] = lastR;
                    dstArray[i + 1] = lastG;
                    dstArray[i + 2] = lastB;
                }
            }
            dstMesh.instanceColor.needsUpdate = true;
        } else if (srcMesh.getColorAt && dstMesh.setColorAt) {
            const tempColor = new THREE.Color();
            for (let i = 0; i < count; i++) {
                srcMesh.getColorAt(i, tempColor);
                dstMesh.setColorAt(i, tempColor);
            }
            if (dstMesh.instanceColor) dstMesh.instanceColor.needsUpdate = true;
        }

        const srcCS = srcMesh.geometry?.getAttribute?.('colorStart');
        const srcCE = srcMesh.geometry?.getAttribute?.('colorEnd');
        const dstCS = dstMesh.geometry?.getAttribute?.('colorStart');
        const dstCE = dstMesh.geometry?.getAttribute?.('colorEnd');
        if (srcCS && srcCE && dstCS && dstCE) {
            const srcStart = srcCS.array;
            const srcEnd = srcCE.array;
            const dstStart = dstCS.array;
            const dstEnd = dstCE.array;
            const copyLen = Math.min(srcStart.length, dstStart.length, count * 3);
            for (let i = 0; i < copyLen; i++) {
                dstStart[i] = srcStart[i];
                dstEnd[i] = srcEnd[i];
            }
            if (dstStart.length > copyLen && copyLen >= 3) {
                const lastSR = dstStart[copyLen - 3];
                const lastSG = dstStart[copyLen - 2];
                const lastSB = dstStart[copyLen - 1];
                const lastER = dstEnd[copyLen - 3];
                const lastEG = dstEnd[copyLen - 2];
                const lastEB = dstEnd[copyLen - 1];
                for (let i = copyLen; i < dstStart.length; i += 3) {
                    dstStart[i] = lastSR;
                    dstStart[i + 1] = lastSG;
                    dstStart[i + 2] = lastSB;
                    dstEnd[i] = lastER;
                    dstEnd[i + 1] = lastEG;
                    dstEnd[i + 2] = lastEB;
                }
            }
            dstCS.needsUpdate = true;
            dstCE.needsUpdate = true;
        }
    }

    // Renamed from updateData. This is for internal data state update only.
    updateDataInternal(newData) {
        // Accept data arrays of any length – the visual component will still use
        // a fixed physical prism count (this.instanceCount).  If fewer data
        // points are supplied than prisms, colours/heights default gracefully.
        if (!newData || !Array.isArray(newData) || newData.length === 0) {
            this.rawData = this.generateTestData();
        } else {
            this.rawData = newData.slice();
        }
        this.normalizedData = this.layerNormalize(this.rawData);
    }

    // Public method for users to update data, which will cause a visual snap.
    // For animation, the PrismLayerNormAnimation class will call updateDataInternal
    // and then manage visual updates via updateInstanceGeometryAndColors(interpolatedHeights).
    updateDataAndSnapVisuals(newData) {
        this.updateDataInternal(newData);
        this._generateKeyColors(); // Regenerate key colors for the snap
        this.updateInstanceGeometryAndColors(); // This applies geometry and default colors
    }
    
    updateColorSubsections(newNumSubsections) {
        this.numSubsections = Math.max(1, Math.floor(newNumSubsections));
        // This will re-calculate colors and also re-apply current heights (animated or static)
        this.updateInstanceGeometryAndColors(); 
    }

    dispose() {
        if (this.mesh.geometry) this.mesh.geometry.dispose();
        if (this.mesh.material) this.mesh.material.dispose();
        this.rawData = [];
        this.normalizedData = [];
        this.instanceUserData = [];
    }

    _generateKeyColors() {
        this.currentKeyColors = [];
        const currentNumSubsections = Math.max(1, this.numSubsections);
        const numKeyColors = currentNumSubsections + 1;
        for (let k = 0; k < numKeyColors; k++) {
            const randomHue = Math.random();
            this.currentKeyColors.push(new THREE.Color().setHSL(randomHue, 1.0, 0.5));
        }
    }

    _updateInstanceColors() {
        if (!this.mesh) {
            console.warn("Mesh not available for color update.");
            return;
        }
        if (this.currentKeyColors.length === 0) {
            // Fallback: Ensure some key colors exist if none were generated
            // This might happen if updateKeyColorsFromData is called before _generateKeyColors
            // or if _generateKeyColors itself had an issue.
            this._generateKeyColors(); 
        }
        // Ensure there are at least two colors for interpolation if numSubsections > 0
        // or at least one if numSubsections is 0 (or 1 key color implies 0 subsections effectively)
        if (this.numSubsections > 0 && this.currentKeyColors.length < 2) {
            console.warn("Not enough key colors for interpolation. Duplicating or defaulting.");
            if (this.currentKeyColors.length === 1) {
                this.currentKeyColors.push(this.currentKeyColors[0].clone()); // Duplicate
            } else { // 0 key colors
                this.currentKeyColors.push(new THREE.Color(0.5, 0.5, 0.5));
                this.currentKeyColors.push(new THREE.Color(0.5, 0.5, 0.5));
            }
        } else if (this.numSubsections === 0 && this.currentKeyColors.length < 1 && this.instanceCount > 0) {
             this.currentKeyColors.push(new THREE.Color(0.5,0.5,0.5)); // Default single color
        }


        const buffers = this._buildColorBuffers();
        this._applyColorBuffers(buffers);
    }

    _buildColorBuffers() {
        const count = this.instanceCount;
        const colorStart = new Float32Array(count * 3);
        const colorEnd = new Float32Array(count * 3);
        const instanceColors = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            const midColor = this.getDefaultColorForIndex(i);
            const leftColor = this.getDefaultColorForIndex(Math.max(0, i - 1));
            const rightColor = this.getDefaultColorForIndex(Math.min(count - 1, i + 1));
            const i3 = i * 3;
            instanceColors[i3] = midColor.r;
            instanceColors[i3 + 1] = midColor.g;
            instanceColors[i3 + 2] = midColor.b;
            colorStart[i3] = leftColor.r;
            colorStart[i3 + 1] = leftColor.g;
            colorStart[i3 + 2] = leftColor.b;
            colorEnd[i3] = rightColor.r;
            colorEnd[i3 + 1] = rightColor.g;
            colorEnd[i3 + 2] = rightColor.b;
        }
        return { colorStart, colorEnd, instanceColors };
    }

    _applyColorBuffers(buffers) {
        if (!buffers) return;
        const colorStartAttr = this.mesh.geometry.getAttribute('colorStart');
        const colorEndAttr = this.mesh.geometry.getAttribute('colorEnd');
        if (colorStartAttr && buffers.colorStart) {
            colorStartAttr.array.set(buffers.colorStart);
            colorStartAttr.needsUpdate = true;
        }
        if (colorEndAttr && buffers.colorEnd) {
            colorEndAttr.array.set(buffers.colorEnd);
            colorEndAttr.needsUpdate = true;
        }
        if (!this.mesh.instanceColor) {
            this.mesh.instanceColor = new THREE.InstancedBufferAttribute(
                new Float32Array(this.instanceCount * 3),
                3
            );
        }
        if (buffers.instanceColors) {
            this.mesh.instanceColor.array.set(buffers.instanceColors);
            this.mesh.instanceColor.needsUpdate = true;
        }
    }

    updateKeyColorsFromData(data, numKeyColorsToSample = 30, colorGenerationOptions = null, cacheKeyData = null) {
        if (!data || data.length === 0) {
            console.warn("No data provided to updateKeyColorsFromData. Using random colors.");
            this.numSubsections = Math.max(1, numKeyColorsToSample -1);
            this._generateKeyColors(); // Generate random colors as a fallback
            this._updateInstanceColors();
            return;
        }

        if (perfStats.enabled) {
            perfStats.inc('colorUpdates');
        }

        const cacheSource = isArrayLike(cacheKeyData) ? cacheKeyData : null;
        const cacheKey = cacheSource
            ? buildColorCacheKey(numKeyColorsToSample, this.instanceCount, colorGenerationOptions)
            : null;
        if (cacheSource && cacheKey) {
            const cached = getColorCacheEntry(cacheSource, cacheKey);
            if (cached && cached.instanceCount === this.instanceCount) {
                this.numSubsections = cached.numSubsections;
                this.currentKeyColors = cached.keyColors;
                this._applyColorBuffers(cached);
                if (perfStats.enabled) {
                    perfStats.inc('colorCacheHits');
                }
                return;
            }
        }

        this.numSubsections = Math.max(0, numKeyColorsToSample - 1); // e.g., 3 samples -> 2 subsections
        this.currentKeyColors = [];
        const dataLengthForSampling = data.length; // Use the length of the provided data array

        if (numKeyColorsToSample === 0) { // No key colors means effectively invisible or uncolored
             this._updateInstanceColors(); // will use default if currentKeyColors is empty
             return;
        }
        
        if (numKeyColorsToSample === 1) {
            const value = dataLengthForSampling > 0 ? data[Math.floor(dataLengthForSampling / 2)] : 0.5; // Use middle value or default
            if (colorGenerationOptions && colorGenerationOptions.type === 'monochromatic') {
                const valueMin = Number.isFinite(colorGenerationOptions.valueMin) ? colorGenerationOptions.valueMin : 0;
                const valueMax = Number.isFinite(colorGenerationOptions.valueMax) ? colorGenerationOptions.valueMax : 1;
                this.currentKeyColors.push(mapValueToMonochromaticColor(
                    value, // This assumes 'value' is somehow normalized or mapValueToMonochromaticColor handles its range
                    colorGenerationOptions.baseHue,
                    colorGenerationOptions.saturation,
                    colorGenerationOptions.minLightness,
                    colorGenerationOptions.maxLightness,
                    valueMin,
                    valueMax
                ));
            } else {
                this.currentKeyColors.push(mapValueToColor(value));
            }
        } else { // numKeyColorsToSample > 1
            const step = dataLengthForSampling > 1 ? (dataLengthForSampling - 1) / (numKeyColorsToSample - 1) : 0;
            for (let i = 0; i < numKeyColorsToSample; i++) {
                const sampleIndex = dataLengthForSampling > 0 ? Math.min(Math.round(i * step), dataLengthForSampling - 1) : 0;
                const value = dataLengthForSampling > 0 ? data[sampleIndex] : 0.5; // Default if no data

                if (colorGenerationOptions && colorGenerationOptions.type === 'monochromatic') {
                     // For monochromatic, the 'value' parameter to mapValueToMonochromaticColor
                     // should probably be the normalized progress (i / (numKeyColorsToSample - 1))
                     // to ensure the gradient spreads across the key colors, rather than raw data values directly dictating lightness.
                     // Or, if raw data should dictate lightness, it needs to be minMaxNormalized first.
                     // Let's assume for now 'value' for monochromatic is progress along the gradient points.
                    const progress = (numKeyColorsToSample > 1) ? i / (numKeyColorsToSample - 1) : 0.5;
                    this.currentKeyColors.push(mapValueToMonochromaticColor(
                        progress, // Use progress for lightness variation
                        colorGenerationOptions.baseHue,
                        colorGenerationOptions.saturation,
                        colorGenerationOptions.minLightness,
                        colorGenerationOptions.maxLightness,
                        0,
                        1
                    ));
                } else {
                    this.currentKeyColors.push(mapValueToColor(value));
                }
            }
        }
        
        // Ensure getDefaultColorForIndex can work (same logic as before)
        if (this.numSubsections === 0 && this.currentKeyColors.length === 0 && numKeyColorsToSample === 1) {
             if (colorGenerationOptions && colorGenerationOptions.type === 'monochromatic') {
                this.currentKeyColors.push(mapValueToMonochromaticColor(
                    0.5,
                    colorGenerationOptions.baseHue,
                    colorGenerationOptions.saturation,
                    colorGenerationOptions.minLightness,
                    colorGenerationOptions.maxLightness,
                    0,
                    1
                ));
            } else {
                this.currentKeyColors.push(new THREE.Color(0.5,0.5,0.5));
            }
        } else if (this.numSubsections > 0 && this.currentKeyColors.length < 2) {
            if (this.currentKeyColors.length === 1) {
                this.currentKeyColors.push(this.currentKeyColors[0].clone()); 
            } else { 
                this.currentKeyColors.push(new THREE.Color(0.5,0.5,0.5));
                this.currentKeyColors.push(new THREE.Color(0.5,0.5,0.5));
            }
        }
        const buffers = this._buildColorBuffers();
        this._applyColorBuffers(buffers);

        if (cacheSource && cacheKey) {
            setColorCacheEntry(cacheSource, cacheKey, {
                instanceCount: this.instanceCount,
                numSubsections: this.numSubsections,
                keyColors: this.currentKeyColors,
                colorStart: buffers.colorStart,
                colorEnd: buffers.colorEnd,
                instanceColors: buffers.instanceColors,
            });
        }
    }

    getDefaultColorForIndex(index) {
        const tempColor = new THREE.Color();
        if (index < 0 || index >= this.instanceCount || this.currentKeyColors.length === 0) {
             return tempColor.setRGB(0.5, 0.5, 0.5); 
        }
        const currentNumSubsections = Math.max(1, this.numSubsections);
        if (this.instanceCount <= 1) {
            tempColor.copy(this.currentKeyColors[0]);
        } else {
            const globalProgress = index / (this.instanceCount - 1);
            const segmentProgress = globalProgress * currentNumSubsections;
            const idx1 = Math.floor(segmentProgress);
            const safeIdx1 = Math.min(idx1, this.currentKeyColors.length - 1);
            const safeIdx2 = Math.min(idx1 + 1, this.currentKeyColors.length - 1);
            const color1 = this.currentKeyColors[safeIdx1];
            const color2 = this.currentKeyColors[safeIdx2];
            const local_t = segmentProgress - idx1;
            tempColor.copy(color1).lerp(color2, local_t);
        }
        return tempColor;
    }

    getUniformHeight() {
        return _uniformCalculatedHeight;
    }

    getWidthScale() { 
        return _prismWidthScale; 
    }

    getDepthScale() { 
        return _prismDepthScale; 
    }

    // Getter for PRISM_BASE_WIDTH (constant from import, but useful for external calculations)
    getBaseWidthConstant() {
        return PRISM_BASE_WIDTH;
    }

    // Helper to override all instances' appearance temporarily (e.g., for a flash or uniform change)
    // Note: yOffset is absolute here, not additive to base position. Scale is absolute.
    overrideAllInstancesTemporary(yPosition, color, uniformScaleVec = null) {
        if (!this.mesh) return;
        const dummy = new THREE.Object3D(); // Re-use a single dummy

        for (let i = 0; i < this.instanceCount; i++) {
            // Base X position for this instance
            const baseX = computeCenteredPrismX(i, this.instanceCount, _prismWidthScale);
            
            // Apply new scale or default scale
            if (uniformScaleVec) {
                dummy.scale.copy(uniformScaleVec);
            } else {
                dummy.scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
            }
            
            // Set new Y position (absolute) and existing X, Z
            // Note: yPosition here is the new center of the prism.
            dummy.position.set(baseX, yPosition, 0); 
            dummy.updateMatrix();
            this.mesh.setMatrixAt(i, dummy.matrix);

            if (color instanceof THREE.Color) {
                this.mesh.setColorAt(i, color);
            }
        }
        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.mesh.instanceColor && color instanceof THREE.Color) {
            this.mesh.instanceColor.needsUpdate = true;
        }
    }

    // Applies final visual state after processing (e.g., shrinking and coloring output vectors)
    applyProcessedVisuals(processedData, numVisibleOutputUnits, colorOptionsForKeyColors, visualOptions = { setHiddenToBlack: true }, cacheKeyData = null) {
        // 1. Update internal rawData with the new (potentially shorter) processedData
        // This rawData will be used by updateKeyColorsFromData for the visible units.
        this.rawData = processedData.slice(); 

        // 2. Generate new key colors based on processedData and colorOptions
        //    The key colors will be applied to the centrally visible prisms.
        const cacheSource = isArrayLike(cacheKeyData) ? cacheKeyData : processedData;
        this.updateKeyColorsFromData(
            this.rawData,
            colorOptionsForKeyColors.numKeyColors,
            colorOptionsForKeyColors.generationOptions,
            cacheSource
        );

        // 3. Determine how many *grouped* prisms should be visible.  Each
        //    prism now represents PRISM_DIMENSIONS_PER_UNIT real dimensions.
        let groupedVisibleUnits = Math.ceil(numVisibleOutputUnits / PRISM_DIMENSIONS_PER_UNIT);
        if (groupedVisibleUnits > this.instanceCount) {
            console.warn(`Grouped visible units (${groupedVisibleUnits}) exceed instanceCount (${this.instanceCount}). Clamping.`);
            groupedVisibleUnits = this.instanceCount;
        }
        const startIndexVisible = Math.floor((this.instanceCount - groupedVisibleUnits) / 2);
        const endIndexVisible = startIndexVisible + groupedVisibleUnits - 1;

        // 4. Set appearance for all physical prisms
        const dummy = new THREE.Object3D();
        for (let i = 0; i < this.instanceCount; i++) {
            const baseX = computeCenteredPrismX(i, this.instanceCount, _prismWidthScale);
            if (i >= startIndexVisible && i <= endIndexVisible) {
                // This is a VISIBLE central prism
                dummy.scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale); // Standard scale
                dummy.position.set(baseX, _uniformCalculatedHeight / 2, 0); // Standard Y position
                
                // Map the processedData index to the current physical prism index `i` for color lookup.
                // The `this.rawData` (now `processedData`) is of length `numVisibleOutputUnits`.
                // `getDefaultColorForIndex` in its current form expects an index from 0 to VECTOR_LENGTH_PRISM-1
                // and interpolates based on `this.currentKeyColors` (which was set up for `numVisibleOutputUnits` length data).
                // We need to map `i` (physical prism) to an effective index into the `processedData` space (0 to numVisibleOutputUnits-1)
                // to get the correct color from the gradient.
                const dataIndexForColor = i - startIndexVisible; // This will be 0 for the first visible prism, up to numVisibleOutputUnits-1
                
                // To use getDefaultColorForIndex, which assumes a globalProgress over VECTOR_LENGTH_PRISM,
                // we need to adjust how it's called or how currentKeyColors were generated.
                // Given currentKeyColors were generated based on `processedData` (length numVisibleOutputUnits)
                // and `this.numSubsections` is set accordingly (numKeyColors-1),
                // getDefaultColorForIndex needs a progress value that makes sense for *that* setup.
                // Let's make getDefaultColorForIndex more flexible or use a specific color getter for this case.

                // Simpler approach for now: Color the visible block with the generated gradient.
                // The current `getDefaultColorForIndex` will use `this.currentKeyColors` (generated from `processedData`)
                // and `this.numSubsections` (based on `colorOptionsForKeyColors.numKeyColors`).
                // It calculates a `globalProgress = index / (VECTOR_LENGTH_PRISM - 1)`. This needs to be rethought.
                // For now, let's assume we want the gradient to span across the *visible* prisms.
                // So, for the first visible prism (i=startIndexVisible), progress should be 0.
                // For the last visible prism (i=endIndexVisible), progress should be 1.
                let progressForColor = 0;
                // Progress along the *visible* portion (0 → 1 across the central prisms)
                if (groupedVisibleUnits > 1) {
                    progressForColor = (i - startIndexVisible) / (groupedVisibleUnits - 1);
                }
                // We need a way to get color based on this progress and `this.currentKeyColors`
                // Let's make a temporary color fetching logic here, or enhance getDefaultColorForIndex
                const colorForVisiblePrism = new THREE.Color();
                if (this.currentKeyColors.length > 0) {
                    if (this.currentKeyColors.length === 1) {
                        colorForVisiblePrism.copy(this.currentKeyColors[0]);
                    } else {
                        const segmentProg = progressForColor * (this.currentKeyColors.length - 1); // numSubsections for colors is keyColors.length - 1
                        const idx1 = Math.floor(segmentProg);
                        const idx2 = Math.min(idx1 + 1, this.currentKeyColors.length - 1);
                        const local_t = segmentProg - idx1;
                        colorForVisiblePrism.copy(this.currentKeyColors[idx1]).lerp(this.currentKeyColors[idx2], local_t);
                    }
                }
                this.mesh.setColorAt(i, colorForVisiblePrism);

            } else {
                // This is an OUTER prism, should be hidden
                dummy.scale.set(0.001, 0.001, 0.001); // Effectively invisible scale
                dummy.position.set(baseX, HIDE_INSTANCE_Y_OFFSET, 0); // Also move far away
                // Only set to black if the option is true (or default)
                if (visualOptions.setHiddenToBlack !== false) {
                    this.mesh.setColorAt(i, new THREE.Color(0,0,0)); // Black for hidden
                }
                // If setHiddenToBlack is explicitly false, their color is not changed by this call,
                // allowing them to retain their current instance color while shrinking/moving.
            }
            dummy.updateMatrix();
            this.mesh.setMatrixAt(i, dummy.matrix);
        }
        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.mesh.instanceColor) {
            this.mesh.instanceColor.needsUpdate = true;
        }
    }
}
