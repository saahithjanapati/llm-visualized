import * as THREE from 'three';

/**
 * Lightweight decorative starfield that orbits around the GPT tower.
 * Stars are positioned above the base of the tower and gently orbit the
 * origin to provide a subtle sense of motion without cluttering the scene.
 */
export class OrbitingStarfield {
    constructor({
        starCount = 18,
        minRadius = 1600,
        maxRadius = 2600,
        baseHeight = 500,
        verticalSpread = 700
    } = {}) {
        this.group = new THREE.Group();
        this.group.name = 'OrbitingStarfield';

        this._enabled = true;
        this._elapsed = 0;
        this._pivots = [];

        const geometry = new THREE.IcosahedronGeometry(22, 0);
        const baseMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.88
        });

        for (let i = 0; i < starCount; i++) {
            const pivot = new THREE.Object3D();
            pivot.rotation.y = Math.random() * Math.PI * 2;

            const radius = THREE.MathUtils.lerp(minRadius, maxRadius, Math.random());
            const height = baseHeight + Math.random() * verticalSpread;
            const scale = 0.45 + Math.random() * 0.4;

            const material = baseMaterial.clone();
            const slightTint = 0.8 + Math.random() * 0.2;
            material.color.setScalar(slightTint);

            const star = new THREE.Mesh(geometry, material);
            star.position.set(radius, height, 0);
            star.scale.setScalar(scale);
            star.renderOrder = 5;

            pivot.add(star);
            this.group.add(pivot);

            this._pivots.push({
                pivot,
                star,
                baseY: height,
                bobAmplitude: 35 + Math.random() * 55,
                bobFrequency: 0.4 + Math.random() * 0.6,
                bobPhase: Math.random() * Math.PI * 2,
                spinSpeed: THREE.MathUtils.degToRad(10 + Math.random() * 30),
                orbitSpeed: (Math.random() * 0.25 + 0.18) * (Math.random() < 0.5 ? 1 : -1)
            });
        }
    }

    setEnabled(enabled) {
        this._enabled = !!enabled;
        this.group.visible = this._enabled;
    }

    update(dt = 0) {
        if (!this._enabled) return;
        if (!Number.isFinite(dt) || dt <= 0) return;
        this._elapsed += dt;

        for (const data of this._pivots) {
            data.pivot.rotation.y += data.orbitSpeed * dt;
            data.star.rotation.y += data.spinSpeed * dt;

            const bob = Math.sin(this._elapsed * data.bobFrequency + data.bobPhase) * data.bobAmplitude;
            data.star.position.y = data.baseY + bob;
        }
    }
}

export default OrbitingStarfield;
