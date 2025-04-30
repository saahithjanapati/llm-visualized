import * as THREE from 'three';
import { mapValueToColor } from '../utils/colors.js';
import { uniformRandom } from '../utils/mathUtils.js';
import { VECTOR_LENGTH, SPHERE_RADIUS, EPSILON, SPHERE_DIAMETER } from '../utils/constants.js';

// Generate a smoother, fully spherical geometry for each bead
const baseSphereGeometry = new THREE.SphereGeometry(SPHERE_RADIUS, 16, 16);
// Preserve the original elongated stone/ellipse look by stretching along the Y and Z axes
const yScale = 4;
const zScale = yScale;
const scaleMatrix = new THREE.Matrix4().makeScale(1, yScale, zScale);
baseSphereGeometry.applyMatrix4(scaleMatrix);

export class VectorNormalizationVisualization {
    constructor(initialPosition = new THREE.Vector3(0, 0, 0)) {
        // Group that contains all visual elements
        this.group = new THREE.Group();
        this.group.position.copy(initialPosition);
        
        // Store original and normalized data
        this.originalData = this.generateTestData();
        this.normalizedData = this.layerNormalize(this.originalData);
        
        // Track animation state
        this.ellipses = [];
        this.unitDelay = 50;     // Delay (ms) between successive unit activations from the center
        this.unitDuration = 1000; // Duration (ms) each unit takes to complete its rise/flash/fall cycle
        this.animationState = {
            isAnimating: false,
            startTime: 0,
            activatedIndices: new Set() // Keep track of which indices have been activated
        };
        this.maxRiseHeight = 4.5;  // Higher peak at center
        this.minRiseHeight = 0.05; // Barely moves at edges
        this.falloffPower = 1.4;  // Exponential fall-off factor (1 = linear)
        this.riseHeight = this.maxRiseHeight; // keep default for compatibility

        // Create vector visualization with original data values
        this.createVectorVisualization();
    }

    generateTestData() {
        const data = [];
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            data.push(uniformRandom() * 4 - 2); // Values between -2 and 2
        }
        return data;
    }

    layerNormalize(vectorData) {
        const sum = vectorData.reduce((acc, val) => acc + val, 0);
        const mean = sum / VECTOR_LENGTH;
        const varianceSum = vectorData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
        const variance = varianceSum / VECTOR_LENGTH;
        const stdDev = Math.sqrt(variance + EPSILON);
        return vectorData.map(val => (val - mean) / stdDev);
    }

    createVectorVisualization() {
        // Create vector visualization with original values
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const value = this.originalData[i];
            const color = mapValueToColor(value);
            const material = new THREE.MeshStandardMaterial({
                color: color,
                metalness: 0.3,
                roughness: 0.5,
                emissive: color,
                emissiveIntensity: 0.3
            });

            const ellipse = new THREE.Mesh(baseSphereGeometry, material);
            ellipse.position.x = (i - VECTOR_LENGTH / 2) * SPHERE_DIAMETER;
            ellipse.position.y = 0; // Initially at rest position
            this.group.add(ellipse);
            this.ellipses.push(ellipse);
        }
    }

    // Start the normalization animation
    startAnimation() {
        this.animationState.isAnimating = true;
        this.animationState.startTime = performance.now();
        this.animationState.activatedIndices = new Set();
    }

    // Update the animation on each frame
    update(currentTime) {
        if (!this.animationState.isAnimating) return;

        const elapsedTime = currentTime - this.animationState.startTime;

        // Determine how far from the center we should have activated by now
        const maxOffset = Math.floor(elapsedTime / this.unitDelay);
        const centerIndex = Math.floor(VECTOR_LENGTH / 2);

        // Activate units outward from the center based on elapsed time
        for (let offset = 0; offset <= maxOffset; offset++) {
            const rightIndex = centerIndex + offset;
            const leftIndex = centerIndex - offset;

            if (rightIndex < VECTOR_LENGTH && !this.animationState.activatedIndices.has(rightIndex)) {
                this.activateUnit(rightIndex);
                this.animationState.activatedIndices.add(rightIndex);
            }
            if (leftIndex >= 0 && !this.animationState.activatedIndices.has(leftIndex)) {
                this.activateUnit(leftIndex);
                this.animationState.activatedIndices.add(leftIndex);
            }
        }

        // Animate all activated units
        let anyUnitStillAnimating = false;
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            if (this.animationState.activatedIndices.has(i)) {
                const stillAnimating = this.animateUnit(i);
                if (stillAnimating) anyUnitStillAnimating = true;
            }
        }

        // Finish the overall animation only when every unit has been activated AND completed
        if (this.animationState.activatedIndices.size === VECTOR_LENGTH && !anyUnitStillAnimating) {
            this.animationState.isAnimating = false;
        }
    }

    // Activate a specific unit to start its animation
    activateUnit(index) {
        const centerIndex = Math.floor(VECTOR_LENGTH / 2);
        const distanceFromCenter = Math.abs(index - centerIndex);
        const activationDelay = distanceFromCenter * this.unitDelay;
        this.ellipses[index].userData.activationTime = this.animationState.startTime + activationDelay;

        // Compute unit-specific peak height: exponential fall-off
        const normalizedDistance = distanceFromCenter / centerIndex; // 0 at center, 1 at edge
        const falloff = Math.pow(1 - normalizedDistance, this.falloffPower); // exponential fall-off
        const peakHeight = this.minRiseHeight + (this.maxRiseHeight - this.minRiseHeight) * falloff;
        this.ellipses[index].userData.riseHeight = peakHeight;

        this.ellipses[index].userData.isAnimating = true;
    }

    // Animate a specific unit that has been activated
    animateUnit(index) {
        const ellipse = this.ellipses[index];
        const activationTime = ellipse.userData.activationTime || 0;
        const now = performance.now();

        if (now < activationTime) return true;

        const unitElapsedTime = now - activationTime;
        const t = Math.min(unitElapsedTime / this.unitDuration, 1); // 0→1 progress

        const peakHeight = ellipse.userData.riseHeight || this.riseHeight;

        // Smooth vertical motion using half-sine (accelerate up, decelerate down)
        ellipse.position.y = peakHeight * Math.sin(Math.PI * t);

        // --- Color / emissive blending ---
        // Create a white flash around the apex (t≈0.5)
        // whitenFactor peaks at 1 at t=0.5 and goes to 0 at t=0 & t=1
        const whitenFactor = Math.sin(Math.PI * t);

        const originalColor = mapValueToColor(this.originalData[index]);
        const normalizedColor = mapValueToColor(this.normalizedData[index]);
        const whiteColor = new THREE.Color(1, 1, 1);

        let tempColor = new THREE.Color();
        if (t < 0.5) {
            // Blend original → white
            const blend = t * 2; // 0→1 between 0 and 0.5
            tempColor.copy(originalColor).lerp(whiteColor, blend);
        } else {
            // Blend white → normalized
            const blend = (t - 0.5) * 2; // 0→1 between 0.5 and 1
            tempColor.copy(whiteColor).lerp(normalizedColor, blend);
        }
        ellipse.material.color.copy(tempColor);
        ellipse.material.emissive.copy(tempColor);
        ellipse.material.emissiveIntensity = 0.3 + 0.7 * whitenFactor;

        if (t >= 1) {
            ellipse.userData.isAnimating = false;
            ellipse.position.y = 0;
            ellipse.material.color.copy(normalizedColor);
            ellipse.material.emissive.copy(normalizedColor);
            ellipse.material.emissiveIntensity = 0.3;
            return false;
        }
        return true;
    }

    // Reset to original state if needed
    reset() {
        this.animationState.isAnimating = false;
        
        // Reset all ellipses to original positions and colors
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const ellipse = this.ellipses[i];
            ellipse.position.y = 0;
            const originalColor = mapValueToColor(this.originalData[i]);
            ellipse.material.color.copy(originalColor);
            ellipse.material.emissive.copy(originalColor);
            ellipse.material.emissiveIntensity = 0.3;
            ellipse.userData.isAnimating = false;
        }
    }

    // Generate new random data and restart
    generateNewData() {
        this.originalData = this.generateTestData();
        this.normalizedData = this.layerNormalize(this.originalData);
        
        // Update original colors
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const value = this.originalData[i];
            const color = mapValueToColor(value);
            this.ellipses[i].material.color.copy(color);
            this.ellipses[i].material.emissive.copy(color);
        }
    }

    dispose() {
        // All ellipses share the same Geometry instance, so we only dispose materials
        this.ellipses.forEach(ellipse => {
            if (ellipse.material) ellipse.material.dispose();
        });
        this.ellipses = [];
    }
} 