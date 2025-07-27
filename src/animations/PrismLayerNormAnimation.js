import * as THREE from 'three';
import { 
    VECTOR_LENGTH_PRISM,
    GROUPED_PRISM_SLOWDOWN
} from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js'; // For color mapping if needed

export class PrismLayerNormAnimation {
    constructor(prismVisualization, config = {}) {
        this.prismVis = prismVisualization;
        this.isAnimating = false;
        this.animationProgress = 0; // Overall animation progress (not currently used, but can be)
        
        const slowdown = GROUPED_PRISM_SLOWDOWN;
        this.config = {
            // Scale delays and durations so total effect length is preserved
            unitDelay: 3 * slowdown,      // ms between activations
            unitDuration: 200 * Math.sqrt(slowdown), // ms per unit cycle
            maxRiseHeight: 12,  // unchanged visual height
            falloffPower: 1.4,   
            minRiseHeight: 2,  
            ...config
        };

        this.activationOrder = [];
        this.unitAnimationStates = []; // To store per-unit animation details
    }

    _calculateActivationOrder() {
        const order = [];
        const centerIndex = Math.floor(VECTOR_LENGTH_PRISM / 2);
        const visited = new Array(VECTOR_LENGTH_PRISM).fill(false);

        for (let offset = 0; ; offset++) {
            let activatedInStep = false;
            const rightIndex = centerIndex + offset;
            const leftIndex = centerIndex - offset;

            if (offset === 0) { // Center element first
                if (rightIndex < VECTOR_LENGTH_PRISM && !visited[rightIndex]) {
                    order.push(rightIndex);
                    visited[rightIndex] = true;
                    activatedInStep = true;
                }
            } else {
                if (rightIndex < VECTOR_LENGTH_PRISM && !visited[rightIndex]) {
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
            if (!activatedInStep && order.length === VECTOR_LENGTH_PRISM) break; // All added
            if (offset > VECTOR_LENGTH_PRISM) break; // Safety break
        }
        return order;
    }

    start(newData) {
        if (!this.prismVis) return;

        this.prismVis.updateDataInternal(newData); // Update underlying data for target colors
        this.isAnimating = true;
        this.animationStartTime = performance.now();
        this.activationOrder = this._calculateActivationOrder();
        
        // Re-generate key colors in prismVis if needed (e.g., if numSubsections changed via GUI before this call)
        // Although updateColorSubsections should handle this, doing it here ensures consistency for this animation run.
        this.prismVis._generateKeyColors(); 

        this.unitAnimationStates = Array(VECTOR_LENGTH_PRISM).fill(null).map((_, index) => {
            const distanceFromCenter = Math.abs(index - Math.floor(VECTOR_LENGTH_PRISM / 2));
            const normalizedDistance = VECTOR_LENGTH_PRISM > 1 ? distanceFromCenter / (VECTOR_LENGTH_PRISM / 2) : 0;
            const falloff = Math.pow(1 - normalizedDistance, this.config.falloffPower);
            const riseHeight = this.config.minRiseHeight + (this.config.maxRiseHeight - this.config.minRiseHeight) * falloff;
            
            return {
                isActive: false,
                activationTime: 0, 
                localProgress: 0,  
                hasCompleted: false,
                riseHeight: riseHeight,
                originalColor: new THREE.Color(), // Will store before flash
                flashTargetColor: mapValueToColor(this.prismVis.normalizedData[index] || 0), // Color based on norm data
                finalRestingColor: this.prismVis.getDefaultColorForIndex(index) // Default subsection gradient color
            };
        });
        
        // Reset all instances to default before starting new animation sequence
        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
            this.prismVis.resetInstanceAppearance(i);
            this.prismVis.mesh.setColorAt(i, this.unitAnimationStates[i].finalRestingColor); // Set initial color explicitly
        }
        this.prismVis.mesh.instanceMatrix.needsUpdate = true;
        this.prismVis.mesh.instanceColor.needsUpdate = true;
    }

    update(deltaTime) {
        if (!this.isAnimating || !this.prismVis) return;

        const currentTime = performance.now();
        const elapsedTimeSinceStart = currentTime - this.animationStartTime;
        let allUnitsStillAnimatingOrPending = false; // Track if any unit is active OR not yet completed

        const activationSlot = Math.floor(elapsedTimeSinceStart / this.config.unitDelay);
        for (let i = 0; i <= activationSlot; i++) {
            if (i < this.activationOrder.length) {
                const unitIndex = this.activationOrder[i];
                if (unitIndex !== undefined && this.unitAnimationStates[unitIndex] && !this.unitAnimationStates[unitIndex].isActive && !this.unitAnimationStates[unitIndex].hasCompleted) {
                    const state = this.unitAnimationStates[unitIndex];
                    state.isActive = true;
                    state.activationTime = currentTime;
                    // Store original color (which should be the finalRestingColor we set in start)
                    state.originalColor.copy(state.finalRestingColor); 
                }
            }
        }

        let needsMatrixUpdate = false;
        let needsColorUpdate = false;

        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
            const state = this.unitAnimationStates[i];
            if (!state) continue; 

            if (state.isActive && !state.hasCompleted) {
                allUnitsStillAnimatingOrPending = true; // Mark that work is still ongoing
                const unitElapsedTime = currentTime - state.activationTime;
                state.localProgress = Math.min(unitElapsedTime / this.config.unitDuration, 1.0);

                const yOffset = state.riseHeight * Math.sin(Math.PI * state.localProgress);

                const tempDisplayColor = new THREE.Color();
                const whiteColor = new THREE.Color(1, 1, 1);
                if (state.localProgress < 0.5) {
                    const t = state.localProgress * 2;
                    tempDisplayColor.copy(state.originalColor).lerp(whiteColor, t);
                } else {
                    const t = (state.localProgress - 0.5) * 2;
                    tempDisplayColor.copy(whiteColor).lerp(state.flashTargetColor, t);
                }

                this.prismVis.setInstanceAppearance(i, yOffset, tempDisplayColor);
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