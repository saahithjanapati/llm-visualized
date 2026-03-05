import * as THREE from 'three';
import { VECTOR_LENGTH_PRISM, PLN_UNIT_DELAY_MS, PLN_UNIT_CYCLE_DURATION_MS, GLOBAL_ANIM_SPEED_MULT } from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js'; // For color mapping if needed

export class PrismLayerNormAnimation {
    constructor(prismVisualization, config = {}) {
        this.prismVis = prismVisualization;
        this.isAnimating = false;
        this.animationProgress = 0; // Overall animation progress (not currently used, but can be)
        this._elapsedMs = 0;
        this._lastWallClockMs = null;
        
        this.config = {
            // Scale delays and durations so total effect length is preserved
            unitDelay: PLN_UNIT_DELAY_MS,      // ms between activations
            unitDuration: PLN_UNIT_CYCLE_DURATION_MS, // ms per unit cycle
            maxRiseHeight: 18,  // increased visual height for stronger rise
            falloffPower: 1.4,
            minRiseHeight: 2,
            ...config
        };

        this.activationOrder = [];
        this.unitAnimationStates = []; // To store per-unit animation details
        this._tempDisplayColor = new THREE.Color();
        this._whiteColor = new THREE.Color(1, 1, 1);
    }

    _nowMs() {
        return (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now()
            : Date.now();
    }

    _calculateActivationOrder(length) {
        const count = Number.isFinite(length) ? Math.max(1, Math.floor(length)) : VECTOR_LENGTH_PRISM;
        const order = [];
        const centerIndex = Math.floor(count / 2);
        const visited = new Array(count).fill(false);

        for (let offset = 0; ; offset++) {
            let activatedInStep = false;
            const rightIndex = centerIndex + offset;
            const leftIndex = centerIndex - offset;

            if (offset === 0) { // Center element first
                if (rightIndex < count && !visited[rightIndex]) {
                    order.push(rightIndex);
                    visited[rightIndex] = true;
                    activatedInStep = true;
                }
            } else {
                if (rightIndex < count && !visited[rightIndex]) {
                    order.push(rightIndex);
                    visited[rightIndex] = true;
                    activatedInStep = true;
                }
                if (leftIndex >= 0 && !visited[leftIndex]) {
                    order.push(leftIndex);
                    visited[leftIndex] = true;
                    activatedInStep = true;
                }
            }
            if (!activatedInStep && order.length === count) break; // All added
            if (offset > count) break; // Safety break
        }
        return order;
    }

    start(newData, options = {}) {
        if (!this.prismVis) return;

        const deferDataUpdate = options && options.deferDataUpdate === true;
        const sourceAlreadyNormalized = options && options.sourceAlreadyNormalized === true;
        const perUnitNormalizedReveal = options && Object.prototype.hasOwnProperty.call(options, 'perUnitNormalizedReveal')
            ? options.perUnitNormalizedReveal !== false
            : deferDataUpdate;
        const dataArray = Array.isArray(newData)
            ? newData
            : ArrayBuffer.isView(newData)
                ? Array.from(newData)
                : [];
        let normalizedData = null;
        if (deferDataUpdate) {
            normalizedData = sourceAlreadyNormalized
                ? dataArray.slice()
                : this.prismVis.layerNormalize(dataArray);
        } else {
            this.prismVis.updateDataInternal(dataArray); // Update underlying data for target colors
            normalizedData = this.prismVis.normalizedData;
        }
        this.isAnimating = true;
        this._elapsedMs = 0;
        this._lastWallClockMs = this._nowMs();
        this._instanceCount = this.prismVis?.instanceCount || VECTOR_LENGTH_PRISM;
        this.activationOrder = this._calculateActivationOrder(this._instanceCount);

        if (!deferDataUpdate) {
            const numKeyColors = Math.min(30, Math.max(1, this.prismVis.rawData.length || 1));
            this.prismVis.updateKeyColorsFromData(this.prismVis.rawData, numKeyColors, null, dataArray);
        }

        this.unitAnimationStates = Array(this._instanceCount).fill(null).map((_, index) => {
            const distanceFromCenter = Math.abs(index - Math.floor(this._instanceCount / 2));
            const normalizedDistance = this._instanceCount > 1 ? distanceFromCenter / (this._instanceCount / 2) : 0;
            const falloff = Math.pow(1 - normalizedDistance, this.config.falloffPower);
            const riseHeight = this.config.minRiseHeight + (this.config.maxRiseHeight - this.config.minRiseHeight) * falloff;
            const startColor = new THREE.Color();
            if (this.prismVis.mesh && typeof this.prismVis.mesh.getColorAt === 'function') {
                this.prismVis.mesh.getColorAt(index, startColor);
            } else {
                startColor.copy(this.prismVis.getDefaultColorForIndex(index));
            }
            
            return {
                isActive: false,
                activationTime: 0, 
                localProgress: 0,  
                hasCompleted: false,
                riseHeight: riseHeight,
                startColor,
                originalColor: new THREE.Color(), // Will store before flash
                flashTargetColor: mapValueToColor((normalizedData && normalizedData[index]) || 0), // Color based on norm data
                finalRestingColor: deferDataUpdate
                    ? mapValueToColor((normalizedData && normalizedData[index]) || 0)
                    : this.prismVis.getDefaultColorForIndex(index) // Default subsection gradient color
            };
        });
        
        // Reset all instances to default before starting new animation sequence
        for (let i = 0; i < this._instanceCount; i++) {
            this.prismVis.resetInstanceAppearance(i);
            const state = this.unitAnimationStates[i];
            const initialColor = perUnitNormalizedReveal
                ? state.startColor
                : state.finalRestingColor;
            this.prismVis.mesh.setColorAt(i, initialColor); // Keep existing color until this unit animates.
        }
        this.prismVis.mesh.instanceMatrix.needsUpdate = true;
        this.prismVis.mesh.instanceColor.needsUpdate = true;
    }

    update(deltaTime) {
        if (!this.isAnimating || !this.prismVis) return;

        const nowMs = this._nowMs();
        let deltaMs = 0;
        if (Number.isFinite(deltaTime)) {
            deltaMs = Math.max(0, deltaTime * 1000);
        } else if (Number.isFinite(this._lastWallClockMs)) {
            deltaMs = Math.max(0, nowMs - this._lastWallClockMs);
        }
        this._lastWallClockMs = nowMs;
        this._elapsedMs += deltaMs;

        const elapsedTimeSinceStart = this._elapsedMs;
        let allUnitsStillAnimatingOrPending = false; // Track if any unit is active OR not yet completed

        const baseSpeedMult = 100;
        const speedScale = baseSpeedMult / Math.max(1, GLOBAL_ANIM_SPEED_MULT);
        const unitDelay = this.config.unitDelay * speedScale;
        const unitDuration = this.config.unitDuration * speedScale;
        const activationSlot = Math.floor(elapsedTimeSinceStart / unitDelay);
        for (let i = 0; i <= activationSlot; i++) {
            if (i < this.activationOrder.length) {
                const unitIndex = this.activationOrder[i];
                if (unitIndex !== undefined && this.unitAnimationStates[unitIndex] && !this.unitAnimationStates[unitIndex].isActive && !this.unitAnimationStates[unitIndex].hasCompleted) {
                    const state = this.unitAnimationStates[unitIndex];
                    state.isActive = true;
                    state.activationTime = elapsedTimeSinceStart;
                    // Start flash from the unit's pre-animation color.
                    state.originalColor.copy(state.startColor);
                }
            }
        }

        let needsMatrixUpdate = false;
        let needsColorUpdate = false;

        for (let i = 0; i < this._instanceCount; i++) {
            const state = this.unitAnimationStates[i];
            if (!state) continue; 

            if (state.isActive && !state.hasCompleted) {
                allUnitsStillAnimatingOrPending = true; // Mark that work is still ongoing
                const unitElapsedTime = elapsedTimeSinceStart - state.activationTime;
                state.localProgress = Math.min(unitElapsedTime / unitDuration, 1.0);

                const yOffset = state.riseHeight * Math.sin(Math.PI * state.localProgress);

                const tempDisplayColor = this._tempDisplayColor;
                const whiteColor = this._whiteColor;
                if (state.localProgress < 0.5) {
                    const t = state.localProgress * 2;
                    tempDisplayColor.copy(state.originalColor).lerp(whiteColor, t);
                } else {
                    const t = (state.localProgress - 0.5) * 2;
                    tempDisplayColor.copy(whiteColor).lerp(state.flashTargetColor, t);
                }

                this.prismVis.setInstanceAppearance(i, yOffset, tempDisplayColor, null, false);
                needsMatrixUpdate = true;
                needsColorUpdate = true;

                if (state.localProgress >= 1.0) {
                    state.hasCompleted = true;
                    state.isActive = false;
                    this.prismVis.resetInstanceAppearance(i); // Reset matrix only
                    this.prismVis.mesh.setColorAt(i, state.finalRestingColor); // Set final color
                    // Matrix/Color already marked true
                }
            } else if (!state.hasCompleted) {
                 allUnitsStillAnimatingOrPending = true; // Unit hasn't completed yet (might not have started)
            }
        }
        
        if(needsMatrixUpdate) this.prismVis.mesh.instanceMatrix.needsUpdate = true;
        if(needsColorUpdate) this.prismVis.mesh.instanceColor.needsUpdate = true;

        // Stop overall animation only when all units have completed their cycle
        if (!allUnitsStillAnimatingOrPending) {
             this.isAnimating = false;
        }
    }
} 
