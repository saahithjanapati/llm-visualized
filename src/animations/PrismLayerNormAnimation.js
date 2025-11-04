import * as THREE from 'three';
import { VECTOR_LENGTH_PRISM, PLN_UNIT_DELAY_MS, PLN_UNIT_CYCLE_DURATION_MS } from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js'; // For color mapping if needed

export class PrismLayerNormAnimation {
    constructor(prismVisualization, config = {}) {
        this.prismVis = prismVisualization;
        this.isAnimating = false;
        this.animationProgress = 0; // Overall animation progress (not currently used, but can be)
        
        this.config = {
            // Scale delays and durations so total effect length is preserved
            unitDelay: PLN_UNIT_DELAY_MS,      // ms between activations
            unitDuration: PLN_UNIT_CYCLE_DURATION_MS, // ms per unit cycle
            maxRiseHeight: 18,  // increased visual height for stronger rise
            falloffPower: 1.4,
            minRiseHeight: 2,
            ...config
        };

        this._tempColor = new THREE.Color();
        this._helperColorA = new THREE.Color();
        this._helperColorB = new THREE.Color();
        this._scaleVec = new THREE.Vector3();

        this._baseWidthScale = (prismVisualization && typeof prismVisualization.getWidthScale === 'function')
            ? prismVisualization.getWidthScale()
            : 1;
        this._baseDepthScale = (prismVisualization && typeof prismVisualization.getDepthScale === 'function')
            ? prismVisualization.getDepthScale()
            : 1;
        this._baseHeight = (prismVisualization && typeof prismVisualization.getUniformHeight === 'function')
            ? prismVisualization.getUniformHeight()
            : 1;

        this.activationOrder = [];
        this.unitAnimationStates = []; // To store per-unit animation details
    }

    _updateBaseScales() {
        if (!this.prismVis) return;
        if (typeof this.prismVis.getWidthScale === 'function') {
            this._baseWidthScale = this.prismVis.getWidthScale();
        }
        if (typeof this.prismVis.getDepthScale === 'function') {
            this._baseDepthScale = this.prismVis.getDepthScale();
        }
        if (typeof this.prismVis.getUniformHeight === 'function') {
            this._baseHeight = this.prismVis.getUniformHeight();
        }
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

    _easeOutBack(t) {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    }

    _computeJoyfulYOffset(state, progress) {
        if (!state) return 0;
        if (progress <= 0) return 0;

        const ascendDuration = 0.45;
        if (progress < ascendDuration) {
            const t = progress / ascendDuration;
            const eased = this._easeOutBack(t);
            const overshoot = 1 + 0.12 * Math.sin(t * Math.PI);
            return state.riseHeight * Math.max(0, eased * overshoot);
        }

        const dropT = (progress - ascendDuration) / (1 - ascendDuration);
        const dropBase = Math.pow(Math.max(0, 1 - dropT), 1.25);
        const wobble = 0.65 + 0.35 * Math.cos(dropT * Math.PI * (1.2 + state.wiggleFrequency * 0.35));
        const sparkle = 0.12 * Math.sin(dropT * Math.PI * 3 + state.pulseOffset);
        const total = Math.max(0, wobble + sparkle);
        return state.riseHeight * dropBase * total;
    }

    _computePlayfulScale(state, progress, timeMs) {
        if (!state) {
            return this._scaleVec.set(this._baseWidthScale, this._baseHeight, this._baseDepthScale);
        }

        const wobble = Math.sin(timeMs * 0.005 + state.wobbleOffset);
        const wiggle = Math.sin(progress * Math.PI * (state.wiggleFrequency + 0.5) + state.pulseOffset);
        const widthPulse = 1 + state.scaleJitter * 0.35 * Math.sin(progress * Math.PI + state.pulseOffset * 0.5);
        const depthPulse = 1 + state.scaleJitter * 0.35 * Math.cos(progress * Math.PI + state.pulseOffset * 0.5);
        const heightPulse = 1 + state.scaleJitter * (0.55 * wiggle + 0.25 * wobble);

        this._scaleVec.set(
            this._baseWidthScale * Math.max(0.1, widthPulse),
            this._baseHeight * Math.max(0.2, heightPulse),
            this._baseDepthScale * Math.max(0.1, depthPulse)
        );
        return this._scaleVec;
    }

    _getPlayfulColor(state, progress, timeMs) {
        if (!state) return this._tempColor.setRGB(1, 1, 1);

        const sparkleOsc = Math.sin(timeMs * 0.006 + state.pulseOffset);

        if (progress < 0.35) {
            const t = progress / 0.35;
            const easedT = THREE.MathUtils.clamp(t * (0.7 + 0.3 * (0.5 + sparkleOsc * 0.5)), 0, 1);
            this._tempColor.copy(state.originalColor).lerp(state.sparkleColor, easedT);
            return this._tempColor;
        }

        if (progress < 0.7) {
            const t = (progress - 0.35) / 0.35;
            const swirl = THREE.MathUtils.clamp(0.35 + 0.65 * Math.sin(timeMs * 0.01 + state.wobbleOffset + t * Math.PI * 2), 0, 1);
            this._helperColorA.copy(state.sparkleColor).lerp(state.highlightColor, THREE.MathUtils.clamp(t, 0, 1));
            this._tempColor.copy(this._helperColorA).lerp(state.flashTargetColor, swirl);
            return this._tempColor;
        }

        const t = THREE.MathUtils.clamp((progress - 0.7) / 0.3, 0, 1);
        this._helperColorB.copy(state.highlightColor).lerp(state.restingBrightColor, t);
        const shimmer = THREE.MathUtils.clamp(0.5 + 0.5 * Math.sin(timeMs * 0.003 + state.pulseOffset), 0, 1);
        const finalMix = (1 - t) * 0.15 * shimmer;
        this._tempColor.copy(this._helperColorB).lerp(state.flashTargetColor, finalMix);
        return this._tempColor;
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
        this._updateBaseScales();

        this.unitAnimationStates = Array(VECTOR_LENGTH_PRISM).fill(null).map((_, index) => {
            const distanceFromCenter = Math.abs(index - Math.floor(VECTOR_LENGTH_PRISM / 2));
            const normalizedDistance = VECTOR_LENGTH_PRISM > 1 ? distanceFromCenter / (VECTOR_LENGTH_PRISM / 2) : 0;
            const falloff = Math.pow(1 - normalizedDistance, this.config.falloffPower);
            const riseHeight = this.config.minRiseHeight + (this.config.maxRiseHeight - this.config.minRiseHeight) * falloff;

            const baseRestColor = this.prismVis.getDefaultColorForIndex(index);
            const finalRestingColor = baseRestColor.clone();
            const flashTargetColor = mapValueToColor(this.prismVis.normalizedData[index] || 0);

            const sparkleHue = (0.5 + Math.random() * 0.35) % 1;
            const accentHue = (sparkleHue + (Math.random() * 0.2 - 0.1) + 1) % 1;
            const sparkleColor = new THREE.Color().setHSL(sparkleHue, 0.85, 0.62);
            const highlightColor = new THREE.Color().setHSL(accentHue, 0.95, 0.72);
            const restingBrightColor = baseRestColor.clone().lerp(flashTargetColor, 0.4).lerp(sparkleColor, 0.18);

            return {
                isActive: false,
                activationTime: 0,
                localProgress: 0,
                hasCompleted: false,
                riseHeight: riseHeight,
                originalColor: baseRestColor.clone(),
                flashTargetColor,
                finalRestingColor,
                sparkleColor,
                highlightColor,
                restingBrightColor,
                pulseOffset: Math.random() * Math.PI * 2,
                wobbleOffset: Math.random() * Math.PI * 2,
                wiggleFrequency: 1.2 + Math.random() * 0.9,
                scaleJitter: 0.05 + Math.random() * 0.05
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

                const yOffset = this._computeJoyfulYOffset(state, state.localProgress);
                const playfulColor = this._getPlayfulColor(state, state.localProgress, currentTime);
                const playfulScale = this._computePlayfulScale(state, state.localProgress, currentTime);

                this.prismVis.setInstanceAppearance(i, yOffset, playfulColor, playfulScale);
                needsMatrixUpdate = true;
                needsColorUpdate = true;

                if (state.localProgress >= 1.0) {
                    state.hasCompleted = true;
                    state.isActive = false;
                    this.prismVis.resetInstanceAppearance(i); // Reset matrix only
                    this.prismVis.mesh.setColorAt(i, state.restingBrightColor); // Set final color
                    needsMatrixUpdate = true;
                    needsColorUpdate = true;
                }
            } else if (!state.hasCompleted) {
                 allUnitsStillAnimatingOrPending = true; // Unit hasn't completed yet (might not have started)
            }
        }
        
        if (needsMatrixUpdate) this.prismVis.mesh.instanceMatrix.needsUpdate = true;
        if (needsColorUpdate) this.prismVis.mesh.instanceColor.needsUpdate = true;

        // Stop overall animation only when all units have completed their cycle
        if (!allUnitsStillAnimatingOrPending) {
             this.isAnimating = false;
        }
    }
} 