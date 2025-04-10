import * as THREE from 'three';
import { VectorVisualization } from '../components/VectorVisualization.js';
import { EmbeddingMatrixVisualization } from '../components/EmbeddingMatrixVisualization.js';
import * as TWEEN from 'three/examples/jsm/libs/tween.module.js'; // Updated path
import { mapValueToColor } from '../utils/colors.js'; // Assuming you have this
import { VECTOR_SPEED, SPAWN_Y } from '../utils/constants.js';

/**
 * Manages the animation sequence of vectors passing through an embedding matrix.
 */
export class MLPAnimation {
    constructor(scene, options = {}) {
        this.scene = scene;
        this.embeddingMatrix = null;
        this.vectors = [];
        this.isActive = false;

        this.options = {
            matrixPosition: new THREE.Vector3(0, 15, 0), // Position the matrix slightly higher
            vectorCount: 5,
            vectorStartY: SPAWN_Y, // Start below the matrix
            vectorSpeed: VECTOR_SPEED * 0.8, // Slightly slower for visibility
            glowColor: new THREE.Color(0xffff00), // Yellow glow
            finalColor: new THREE.Color(0x8B8000), // Darker yellow
            activationDuration: 1500, // ms for matrix to light up
            fadeDuration: 2000, // ms for matrix to fade after activation
            ...options
        };

        this._setup();
    }

    _setup() {
        // Create the Embedding Matrix
        this.embeddingMatrix = new EmbeddingMatrixVisualization({
            position: this.options.matrixPosition,
            slitCount: this.options.vectorCount,
            bottomWidth: 25, // Adjust as needed
            topWidth: 20,
            height: 6,
            depth: 3
        });
        this.scene.add(this.embeddingMatrix.group);

        // Prepare vectors (but don't add to scene yet)
        const slitPositions = this.embeddingMatrix.slitPositions;
        for (let i = 0; i < this.options.vectorCount; i++) {
             const initialPosition = new THREE.Vector3(
                this.options.matrixPosition.x + slitPositions[i],
                this.options.vectorStartY,
                this.options.matrixPosition.z // Align Z with matrix center
            );
            // Create vector with random initial data
            const randomData = Array.from({ length: 5 }, () => Math.random() * 2 - 1); // Example dim 5, values -1 to 1
            const vectorVis = new VectorVisualization(randomData, initialPosition);
            vectorVis.speed = this.options.vectorSpeed;
            vectorVis.isActive = false; // Mark as inactive until animation starts
            this.vectors.push(vectorVis);
        }
    }

    start() {
        if (this.isActive) return;
        this.isActive = true;

        console.log("Starting MLP Animation");

        // Add vectors to the scene
        this.vectors.forEach(v => {
            this.scene.add(v.group);
            v.isActive = true;
        });

        // Animation for the embedding matrix color/glow
        const matrixMaterial = this.embeddingMatrix.material;
        const initialColor = this.embeddingMatrix.initialColor.clone();

        // Tween to bright yellow glow
        new TWEEN.Tween(matrixMaterial.emissive)
            .to({ r: this.options.glowColor.r, g: this.options.glowColor.g, b: this.options.glowColor.b }, this.options.activationDuration)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                 matrixMaterial.emissiveIntensity = 1.5; // Make it glow brighter during activation
             })
            .start()
            .onComplete(() => {
                console.log("Matrix activated (yellow glow)");
                // After glowing, tween the main color and fade emissive
                new TWEEN.Tween(matrixMaterial.color)
                    .to({ r: this.options.finalColor.r, g: this.options.finalColor.g, b: this.options.finalColor.b }, this.options.fadeDuration)
                    .easing(TWEEN.Easing.Linear.None)
                    .start();

                new TWEEN.Tween(matrixMaterial.emissive)
                    .to({ r: this.options.finalColor.r, g: this.options.finalColor.g, b: this.options.finalColor.b }, this.options.fadeDuration) // Fade emissive towards final color
                    .easing(TWEEN.Easing.Linear.None)
                    .onUpdate(() => {
                        // Gradually reduce intensity as it fades
                        matrixMaterial.emissiveIntensity = THREE.MathUtils.lerp(1.5, 0.5, TWEEN.Interpolation.Utils.Linear(0, 1));
                    })
                    .start()
                     .onComplete(() => {
                        console.log("Matrix faded to darker yellow");
                        matrixMaterial.emissiveIntensity = 0.5; // Keep a dim emissive matching the final color
                     });
            });

    }

    update(deltaTime) {
        if (!this.isActive) return;

        TWEEN.update(); // Update all active tweens

        const matrixTopY = this.embeddingMatrix.group.position.y + this.embeddingMatrix.height;
        const matrixBottomY = this.embeddingMatrix.group.position.y;

        // Animate vectors passing through
        for (let i = this.vectors.length - 1; i >= 0; i--) {
            const vectorVis = this.vectors[i];

            if (!vectorVis.isActive) continue; // Skip inactive vectors

            const previousY = vectorVis.group.position.y;
            vectorVis.group.position.y += vectorVis.speed; // Move vector
            const currentY = vectorVis.group.position.y;

            // Check if the vector just passed *through* the matrix center height
            if (previousY < matrixBottomY + this.embeddingMatrix.height / 2 && currentY >= matrixBottomY + this.embeddingMatrix.height / 2) {
                console.log(`Vector ${i} passing through matrix`);
                // Modify vector data (example: randomize again)
                const newRandomData = Array.from({ length: vectorVis.dimension }, () => Math.random() * 2 - 1);
                vectorVis.updateData(newRandomData);
            }

            // Check if vector is way past the matrix to potentially remove or stop
            // For now, let them continue upwards until handled by main loop's despawn logic
            // if (currentY > matrixTopY + 50) { // Example threshold
                 // vectorVis.isActive = false; // Mark as done with this animation
                 // this.scene.remove(vectorVis.group);
                 // vectorVis.dispose();
                 // this.vectors.splice(i, 1);
            // }
        }

        // Optional: Check if all vectors have passed and potentially reset or stop the animation
        // const allPassed = this.vectors.every(v => v.group.position.y > matrixTopY + 10);
        // if (allPassed && this.vectors.length > 0) {
        //     console.log("MLP Animation sequence likely complete.");
             // this.isActive = false; // Or trigger cleanup/next stage
        // }
    }

    dispose() {
        this.isActive = false;
        TWEEN.removeAll(); // Stop all tweens associated with this animation

        if (this.embeddingMatrix) {
            this.scene.remove(this.embeddingMatrix.group);
            this.embeddingMatrix.dispose();
            this.embeddingMatrix = null;
        }
        this.vectors.forEach(vectorVis => {
             if (vectorVis.group.parent === this.scene) {
                 this.scene.remove(vectorVis.group);
             }
             vectorVis.dispose();
        });
        this.vectors = [];
        console.log("MLP Animation disposed");
    }
}
