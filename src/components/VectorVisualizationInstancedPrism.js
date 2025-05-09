import * as THREE from 'three';
import { 
    VECTOR_LENGTH_PRISM, 
    PRISM_BASE_WIDTH,
    PRISM_BASE_DEPTH,
    PRISM_MAX_HEIGHT,
    PRISM_HEIGHT_SCALE_FACTOR,
} from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js';

// Base geometry has height 1, will be scaled by data
const basePrismGeometry = new THREE.BoxGeometry(PRISM_BASE_WIDTH, 1, PRISM_BASE_DEPTH);

// Pre-calculate fixed uniform height and scales for the prisms
let _uniformCalculatedHeight = PRISM_MAX_HEIGHT * PRISM_HEIGHT_SCALE_FACTOR * 2.0; // Double height
_uniformCalculatedHeight = Math.max(0.01, _uniformCalculatedHeight); // Ensure minimum
const _prismWidthScale = 1.5;
const _prismDepthScale = 1.5;

export class VectorVisualizationInstancedPrism {
    constructor(initialData = null, initialPosition = new THREE.Vector3(0, 0, 0), numSubsections = 30) {
        this.group = new THREE.Group();
        this.group.position.copy(initialPosition);

        this.rawData = initialData || this.generateTestData();
        this.normalizedData = []; // Will be populated by updateDataInternal
        this.numSubsections = numSubsections;
        this.currentKeyColors = []; // To store the current random key colors for subsections

        // For storing individual instance animation states if needed by an external controller
        this.instanceUserData = Array(VECTOR_LENGTH_PRISM).fill(null).map(() => ({})); 

        const material = new THREE.MeshBasicMaterial({ 
            color: new THREE.Color(0xffffff) // Base color for material, instance colors will override
        });

        const instancedGeometry = basePrismGeometry.clone();
        this.mesh = new THREE.InstancedMesh(instancedGeometry, material, VECTOR_LENGTH_PRISM);
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this.group.add(this.mesh);
        
        this.updateDataInternal(this.rawData.slice()); // Process initial data
        this._generateKeyColors(); // Generate initial key colors
        this.updateInstanceGeometryAndColors();      // Set initial visual state (fixed dimensions, subsection colors)
    }

    generateTestData() {
        const data = [];
        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
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

        // Use the stored key colors
        if (this.currentKeyColors.length === 0) this._generateKeyColors();

        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
            // Apply fixed, uniform dimensions
            const x = (i - VECTOR_LENGTH_PRISM / 2) * (PRISM_BASE_WIDTH * _prismWidthScale);
            dummy.scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
            dummy.position.set(x, _uniformCalculatedHeight / 2, 0); 
            dummy.updateMatrix();
            this.mesh.setMatrixAt(i, dummy.matrix);

            // Set color using the stored key colors and getDefaultColorForIndex logic
            this.mesh.setColorAt(i, this.getDefaultColorForIndex(i));
        }

        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.mesh.instanceColor) {
            this.mesh.instanceColor.needsUpdate = true;
        } else {
            // This warning should ideally not appear with correct material setup
            console.warn("instanceColor buffer does NOT exist on the mesh after setColorAt!");
        }
    }

    // Updates the visual appearance of a single instance for animation purposes
    setInstanceAppearance(index, yOffset, tempColor) {
        if (index < 0 || index >= VECTOR_LENGTH_PRISM) return;
        const currentMatrix = new THREE.Matrix4();
        this.mesh.getMatrixAt(index, currentMatrix);
        const position = new THREE.Vector3();
        const quaternion = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        currentMatrix.decompose(position, quaternion, scale);
        const baseX = (index - VECTOR_LENGTH_PRISM / 2) * (PRISM_BASE_WIDTH * _prismWidthScale);
        const basePrismYPos = _uniformCalculatedHeight / 2;
        position.set(baseX, basePrismYPos + yOffset, 0);
        
        // Explicitly set the correct scale during compose
        scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);

        currentMatrix.compose(position, quaternion, scale);
        this.mesh.setMatrixAt(index, currentMatrix);
        this.mesh.instanceMatrix.needsUpdate = true;

        if (tempColor instanceof THREE.Color) {
            this.mesh.setColorAt(index, tempColor);
            if (this.mesh.instanceColor) {
                this.mesh.instanceColor.needsUpdate = true;
            }
        }
    }

    // Resets a single instance to its default appearance (fixed dimensions, subsection color)
    resetInstanceAppearance(index) {
        if (index < 0 || index >= VECTOR_LENGTH_PRISM) return;
        
        const dummy = new THREE.Object3D();
        const x = (index - VECTOR_LENGTH_PRISM / 2) * (PRISM_BASE_WIDTH * _prismWidthScale);
        dummy.scale.set(_prismWidthScale, _uniformCalculatedHeight, _prismDepthScale);
        dummy.position.set(x, _uniformCalculatedHeight / 2, 0); 
        dummy.updateMatrix();
        this.mesh.setMatrixAt(index, dummy.matrix);
    }

    // Renamed from updateData. This is for internal data state update only.
    updateDataInternal(newData) {
        if (!newData || !Array.isArray(newData) || newData.length !== VECTOR_LENGTH_PRISM) {
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
        this.updateInstanceGeometryAndColors();
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
        if (!this.mesh || !this.mesh.instanceColor) {
            console.warn("Mesh or instanceColor buffer not available for color update.");
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
        } else if (this.numSubsections === 0 && this.currentKeyColors.length < 1 && VECTOR_LENGTH_PRISM > 0) {
             this.currentKeyColors.push(new THREE.Color(0.5,0.5,0.5)); // Default single color
        }


        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
            this.mesh.setColorAt(i, this.getDefaultColorForIndex(i));
        }
        this.mesh.instanceColor.needsUpdate = true;
    }

    updateKeyColorsFromData(data, numKeyColorsToSample = 30) {
        if (!data || data.length === 0) {
            console.warn("No data provided to updateKeyColorsFromData. Using random colors.");
            this.numSubsections = Math.max(1, numKeyColorsToSample -1);
            this._generateKeyColors(); // Generate random colors as a fallback
            this._updateInstanceColors();
            return;
        }

        this.numSubsections = Math.max(1, numKeyColorsToSample - 1); // e.g., 30 samples -> 29 subsections
        this.currentKeyColors = [];

        if (numKeyColorsToSample === 1) { // Single key color desired
            const value = data[Math.floor(data.length / 2)]; // Use middle value
            this.currentKeyColors.push(mapValueToColor(value));
        } else {
            const step = (data.length -1) / (numKeyColorsToSample - 1);
            for (let i = 0; i < numKeyColorsToSample; i++) {
                const sampleIndex = Math.min(Math.round(i * step), data.length - 1);
                const value = data[sampleIndex];
                this.currentKeyColors.push(mapValueToColor(value));
            }
        }
        
        // Ensure getDefaultColorForIndex can work: needs at least 1 color if numSubsections is 0 (implies 1 color),
        // or 2 colors if numSubsections > 0 for lerp.
        if (this.numSubsections === 0 && this.currentKeyColors.length === 0 && numKeyColorsToSample === 1) {
            // This case should be covered by the numKeyColorsToSample === 1 block.
            // If still empty, add a default.
             this.currentKeyColors.push(new THREE.Color(0.5,0.5,0.5));
        } else if (this.numSubsections > 0 && this.currentKeyColors.length < 2) {
            if (this.currentKeyColors.length === 1) {
                this.currentKeyColors.push(this.currentKeyColors[0].clone()); // Duplicate if only one for lerp
            } else { // 0 key colors after sampling (e.g. data was empty before)
                this.currentKeyColors.push(new THREE.Color(0.5,0.5,0.5));
                this.currentKeyColors.push(new THREE.Color(0.5,0.5,0.5));
            }
        }
        this._updateInstanceColors();
    }

    getDefaultColorForIndex(index) {
        const tempColor = new THREE.Color();
        if (index < 0 || index >= VECTOR_LENGTH_PRISM || this.currentKeyColors.length === 0) {
             return tempColor.setRGB(0.5, 0.5, 0.5); 
        }
        const currentNumSubsections = Math.max(1, this.numSubsections);
        if (VECTOR_LENGTH_PRISM <= 1) {
            tempColor.copy(this.currentKeyColors[0]);
        } else {
            const globalProgress = index / (VECTOR_LENGTH_PRISM - 1);
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
}