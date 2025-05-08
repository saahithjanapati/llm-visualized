import * as THREE from 'three';
// Use the global TWEEN imported in HTML via import map
import { 
    VECTOR_LENGTH_PRISM,
    HIDE_INSTANCE_Y_OFFSET,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_BASE_Y_OFFSET_FACTOR
} from '../utils/constants.js';
import { mapValueToColor } from '../utils/colors.js';

export class PrismAdditionAnimation {
    constructor(sourceVis, targetVis, config = {}) {
        this.sourceVis = sourceVis;
        this.targetVis = targetVis;
        this.isAnimating = false;

        this.config = {
            duration: PRISM_ADD_ANIM_BASE_DURATION,          // MODIFIED
            flashDuration: PRISM_ADD_ANIM_BASE_FLASH_DURATION, // MODIFIED
            delayBetweenPrisms: PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS, // MODIFIED
            yOffsetFactor: PRISM_ADD_ANIM_BASE_Y_OFFSET_FACTOR,     // MODIFIED
            ...config
        };
    }

    start() {
        if (!this.sourceVis || !this.targetVis || this.isAnimating) return;

        const TWEEN = window.TWEEN;
        console.log('PrismAdditionAnimation: window.TWEEN === imported TWEEN?', typeof TWEEN.update !== 'undefined');
        console.log('PrismAdditionAnimation: initial tween count', TWEEN.getAll ? TWEEN.getAll().length : 'no getAll');
        if (!TWEEN) {
            console.error("PrismAdditionAnimation Error: Global TWEEN instance not available. Make sure to set window.TWEEN before using the animation.");
            return;
        }

        if (typeof this.sourceVis.getUniformHeight !== 'function') {
            console.error("PrismAdditionAnimation Error: sourceVis missing getUniformHeight() method.");
            return;
        }

        const uniformPrismHeight = this.sourceVis.getUniformHeight();
        this.isAnimating = true;
        TWEEN.removeAll(); // Clear any previous tweens
        console.log('PrismAdditionAnimation: After explicit removeAll, count:', TWEEN.getAll ? TWEEN.getAll().length : 'no getAll');

        // --- Test with a single, very simple tween ---
        const simpleTestData = { x: 0 };
        const simpleTween = new TWEEN.Tween(simpleTestData)
            .to({ x: 1 }, 100) // Short duration, simple target
            .onComplete(() => console.log('Simple test tween completed.'))
            .start();
        
        const countAfterSimpleTween = TWEEN.getAll ? TWEEN.getAll().length : -1;
        console.log('PrismAdditionAnimation: Count AFTER starting one simple test tween:', countAfterSimpleTween);

        if (countAfterSimpleTween === 0) {
            console.error('CRITICAL TWEEN ISSUE: Starting a simple tween did NOT add it to TWEEN.getAll(). The TWEEN library might not be functioning correctly or is being interfered with.');
            console.log('Simple test tween object:', simpleTween);
            this.isAnimating = false; // Prevent further execution
            return; // Exit early
        }
        // --- End of simple tween test ---

        // --- Restore Original TWEEN Logic ---
        const sourceBaseY = this.sourceVis.group.position.y + uniformPrismHeight / 2;
        const targetBaseY = this.targetVis.group.position.y + uniformPrismHeight / 2;
        const targetYOffset = targetBaseY - sourceBaseY;

        const totalDuration = this.config.duration + this.config.flashDuration + (VECTOR_LENGTH_PRISM - 1) * this.config.delayBetweenPrisms;
        console.log(`Starting Addition Animation (Restored TWEEN). Expected duration: ${totalDuration}ms`);
        console.log(`Source Group Y: ${this.sourceVis.group.position.y}, Target Group Y: ${this.targetVis.group.position.y}`);
        console.log(`Uniform Prism Height: ${uniformPrismHeight}`);
        console.log(`Source Base Y: ${sourceBaseY}, Target Base Y: ${targetBaseY}`);
        console.log(`Calculated targetYOffset for tween: ${targetYOffset}`);

        if (isNaN(targetYOffset)) {
            console.error("PrismAdditionAnimation Error: targetYOffset is NaN. Check getUniformHeight() or group positions.");
            this.isAnimating = false;
            return;
        }

        const finalTargetRawData = this.targetVis.rawData.slice();
        let activeTweens = VECTOR_LENGTH_PRISM; // Count down tweens

        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
            const startYOffset = 0; 
            const tweenData = { yOffset: startYOffset };

            const moveTween = new TWEEN.Tween(tweenData)
                .to({ yOffset: targetYOffset }, this.config.duration)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .delay(i * this.config.delayBetweenPrisms)
                .onUpdate(() => {
                    if (i === 0) { // Log only for the first prism to avoid console spam
                        console.log(`Tween onUpdate for prism ${i}: yOffset = ${tweenData.yOffset.toFixed(2)}, matrix exists: ${!!(this.sourceVis && this.sourceVis.mesh && this.sourceVis.mesh.instanceMatrix)}`);
                    }
                    this.sourceVis.setInstanceAppearance(i, tweenData.yOffset, null);
                    // instanceMatrix.needsUpdate is set inside setInstanceAppearance
                })
                .onComplete(() => {
                    // 1. Update the targetVis's rawData and get its final gradient color (index-based)
                    this.targetVis.rawData[i] = finalTargetRawData[i] + this.sourceVis.rawData[i];
                    finalTargetRawData[i] = this.targetVis.rawData[i]; // Accumulate for final full update
                    const finalGradientColor = this.targetVis.getDefaultColorForIndex(i);

                    // 2. Make the source prism adopt this gradient color at arrival position.
                    this.sourceVis.setInstanceAppearance(i, targetYOffset, finalGradientColor);

                    // 3. Set target prism to its gradient color.
                    this.targetVis.setInstanceAppearance(i, 0, finalGradientColor);

                    // 4. Hide source prism
                    this.sourceVis.setInstanceAppearance(i, HIDE_INSTANCE_Y_OFFSET, null);
                    
                    activeTweens--;
                    if (activeTweens === 0) {
                        console.log("Addition Animation: All tweens completed.");
                        this.targetVis.updateDataInternal(finalTargetRawData);
                        this.targetVis.updateInstanceGeometryAndColors(); // restore gradient across all prisms
                        this.isAnimating = false;
                    }
                });

            moveTween.start();
        }
        console.log('PrismAdditionAnimation: tween count AFTER starting all tweens', TWEEN.getAll ? TWEEN.getAll().length : 'no getAll');
    }

    update(deltaTime) {
        if (!this.isAnimating) return; 

        // TWEEN.update(); // REMOVED - Main animation loop in HTML handles this

        // After tweens run and potentially call setInstanceAppearance, 
        // mark the meshes as needing updates for this frame.
        if (this.sourceVis && this.sourceVis.mesh) {
            if (this.sourceVis.mesh.instanceMatrix) this.sourceVis.mesh.instanceMatrix.needsUpdate = true;
            if (this.sourceVis.mesh.instanceColor) this.sourceVis.mesh.instanceColor.needsUpdate = true; // Mark color too, just in case
        }
        if (this.targetVis && this.targetVis.mesh) {
            if (this.targetVis.mesh.instanceMatrix) this.targetVis.mesh.instanceMatrix.needsUpdate = true;
            if (this.targetVis.mesh.instanceColor) this.targetVis.mesh.instanceColor.needsUpdate = true;
        }
        
        // isAnimating flag is set to false internally by the last TWEEN onComplete callback
    }
} 