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
            wiggleFrequency: 3.2,
            wiggleAmplitude: 0.25,
            wigglePhaseSpacing: 0.8,
            scalePulseStrength: 0.1,
            paletteColors: [0xff8bd7, 0xfff59e, 0x8ef7ff, 0xc7a8ff, 0xffc0a9],
            paletteInfluence: 0.65,
            palettePhaseOffset: 0.16,
            colorOrbitSpeed: 0.0011,
            ...config
        };

        this._white = new THREE.Color(1, 1, 1);
        this._paletteColorScratch = new THREE.Color();
        this._scaleScratch = new THREE.Vector3();
        this._defaultScale = new THREE.Vector3(1, 1, 1);
        this._refreshPaletteColors();
        this._updateDefaultScaleFromPrism();

        this.activationOrder = [];
        this.unitAnimationStates = []; // To store per-unit animation details
    }

    _refreshPaletteColors() {
        const palette = Array.isArray(this.config.paletteColors) ? this.config.paletteColors : [];
        this._paletteColors = palette.map((entry) => {
            if (entry instanceof THREE.Color) {
                return entry.clone();
            }
            if (typeof entry === 'number') {
                return new THREE.Color(entry);
            }
            if (Array.isArray(entry) && entry.length === 3) {
                return new THREE.Color().setRGB(entry[0], entry[1], entry[2]);
            }
            return new THREE.Color(1, 1, 1);
        });
        if (!this._paletteColors.length) {
            this._paletteColors.push(this._white.clone());
        }
    }

    _sampleJoyfulPalette(t) {
        if (!this._paletteColors || !this._paletteColors.length) {
            return this._paletteColorScratch.copy(this._white);
        }
        const wrapped = ((t % 1) + 1) % 1;
        const scaled = wrapped * this._paletteColors.length;
        const idx = Math.floor(scaled) % this._paletteColors.length;
        const nextIdx = (idx + 1) % this._paletteColors.length;
        const lerpT = scaled - Math.floor(scaled);
        return this._paletteColorScratch.copy(this._paletteColors[idx]).lerp(this._paletteColors[nextIdx], lerpT);
    }

    _updateDefaultScaleFromPrism() {
        if (!this.prismVis) {
            this._defaultScale.set(1, 1, 1);
            return;
        }
        const width = typeof this.prismVis.getWidthScale === 'function' ? this.prismVis.getWidthScale() : 1;
        const height = typeof this.prismVis.getUniformHeight === 'function' ? this.prismVis.getUniformHeight() : 1;
        const depth = typeof this.prismVis.getDepthScale === 'function' ? this.prismVis.getDepthScale() : 1;
        this._defaultScale.set(width, height, depth);
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
        this._refreshPaletteColors();
        this._updateDefaultScaleFromPrism();

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
                finalRestingColor: this.prismVis.getDefaultColorForIndex(index), // Default subsection gradient color
                distanceFactor: falloff,
                palettePhase: (index * this.config.palettePhaseOffset) % 1,
                wiggleSeed: index * this.config.wigglePhaseSpacing
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

                const baseBounce = Math.sin(Math.PI * state.localProgress);
                const wiggle = Math.sin(state.localProgress * Math.PI * this.config.wiggleFrequency + state.wiggleSeed);
                const joyfulBounce = Math.max(0, baseBounce + this.config.wiggleAmplitude * baseBounce * wiggle);
                const yOffset = state.riseHeight * joyfulBounce;

                const palettePhase = ((currentTime - this.animationStartTime) * this.config.colorOrbitSpeed + state.palettePhase) % 1;
                const paletteColor = this._sampleJoyfulPalette(palettePhase);

                const tempDisplayColor = new THREE.Color();
                tempDisplayColor.copy(state.originalColor)
                    .lerp(this._white, 0.4 * baseBounce)
                    .lerp(state.flashTargetColor, 0.45 + 0.45 * baseBounce)
                    .lerp(paletteColor, this.config.paletteInfluence * joyfulBounce);

                const scalePulse = 1 + this.config.scalePulseStrength * state.distanceFactor * baseBounce;
                const newScale = this._scaleScratch.copy(this._defaultScale);
                newScale.y *= scalePulse;

                this.prismVis.setInstanceAppearance(i, yOffset, tempDisplayColor, newScale);
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
