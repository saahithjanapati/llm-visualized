import * as THREE from 'three';

/**
 * Simple addition animation between two VectorVisualization32 instances (now supporting any group count).
 * Each of the mesh components from the source vector travels vertically
 * upward until it overlaps the corresponding component in the target vector,
 * then disappears (simulating the values being added).
 */
export class GroupedVectorAdditionAnimation {
    constructor(sourceVis, targetVis, config = {}) {
        this.sourceVis = sourceVis;
        this.targetVis = targetVis;
        this.isAnimating = false;

        this.config = {
            duration: 400, // ms per prism travel
            delayBetween: 15, // stagger between components
            ...config
        };
    }

    start() {
        if (this.isAnimating || !this.sourceVis || !this.targetVis) return;
        const TWEEN = window.TWEEN;
        if (!TWEEN) {
            console.error('GroupedVectorAdditionAnimation: TWEEN (global) not found.');
            return;
        }

        const yOffset = this.targetVis.group.position.y - this.sourceVis.group.position.y;
        this.isAnimating = true;
        const componentCount = this.sourceVis.group.children.length;
        let remaining = componentCount;

        for (let i = 0; i < componentCount; i++) {
            const mesh = this.sourceVis.group.children[i];
            if (!mesh) continue;
            const startY = mesh.position.y;
            const tweenData = { y: startY };

            new TWEEN.Tween(tweenData)
                .to({ y: startY + yOffset }, this.config.duration)
                .delay(i * this.config.delayBetween)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    mesh.position.y = tweenData.y;
                })
                .onComplete(() => {
                    mesh.visible = false; // hide the source component

                    // Flash effect on the corresponding target mesh
                    const tgtMesh = this.targetVis.group.children[i];
                    if (tgtMesh) {
                        // Create a white overlay mesh slightly larger so it bleeds
                        const flashGeo  = tgtMesh.geometry.clone();
                        const flashMat  = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 1.0 });
                        const flashMesh = new THREE.Mesh(flashGeo, flashMat);
                        flashMesh.position.copy(tgtMesh.position);
                        flashMesh.scale.copy(tgtMesh.scale).multiplyScalar(1.05); // slightly larger
                        this.targetVis.group.add(flashMesh);

                        // Fade out quickly
                        new TWEEN.Tween(flashMat)
                            .to({ opacity: 0 }, 200)
                            .easing(TWEEN.Easing.Quadratic.Out)
                            .onComplete(() => {
                                this.targetVis.group.remove(flashMesh);
                                flashGeo.dispose();
                                flashMat.dispose();
                            })
                            .start();
                    }

                    if (--remaining === 0) {
                        this.isAnimating = false;
                    }
                })
                .start();
        }
    }

    update() {
        // Nothing extra; position updates already applied in tween callbacks
    }
} 