import * as THREE from 'three';

/**
 * Minimal orbiting star field that circles the GPT tower at a fixed height.
 * Stars are represented as tiny glowing icosahedrons with subtle rotation.
 */
export class OrbitingStars {
    constructor(scene, options = {}) {
        this._scene = scene;
        this._isEnabled = false;
        this.group = new THREE.Group();
        this.group.name = 'OrbitingStars';
        this.group.visible = false;

        const count = Math.max(4, Math.floor(options.count ?? 14));
        const baseRadius = options.radius ?? 260;
        const radiusJitter = options.radiusJitter ?? 60;
        const baseHeight = options.height ?? 120;
        const heightJitter = options.heightJitter ?? 25;
        const baseSpeed = options.baseAngularSpeed ?? 0.35;
        const speedJitter = options.speedJitter ?? 0.2;
        const spinSpeed = options.spinSpeed ?? 0.6;

        const palette = [0xffffff, 0xcde5ff, 0xfff3c1];
        this._sharedGeometry = new THREE.IcosahedronGeometry(1, 0);

        this._stars = [];
        for (let i = 0; i < count; i++) {
            const color = palette[i % palette.length];
            const material = new THREE.MeshBasicMaterial({
                color,
                toneMapped: false
            });
            const mesh = new THREE.Mesh(this._sharedGeometry, material);
            const scale = (options.minSize ?? 1.2) + Math.random() * ((options.maxSize ?? 2.6) - (options.minSize ?? 1.2));
            mesh.scale.setScalar(scale);
            mesh.position.y = baseHeight;
            mesh.layers.enable(0);

            const angle = Math.random() * Math.PI * 2;
            const radius = baseRadius + (Math.random() * 2 - 1) * radiusJitter;
            const heightOffset = (Math.random() * 2 - 1) * heightJitter;
            const angularSpeed = baseSpeed + (Math.random() * 2 - 1) * speedJitter;
            const spin = (Math.random() * 2 - 1) * spinSpeed;

            this._stars.push({ mesh, material, angle, radius, heightOffset, angularSpeed, spin });
            this.group.add(mesh);
        }

        if (this._scene) {
            this._scene.add(this.group);
        }

        this._baseHeight = baseHeight;
    }

    setEnabled(enabled) {
        this._isEnabled = !!enabled;
        if (this.group) {
            this.group.visible = this._isEnabled;
        }
    }

    update(deltaSeconds) {
        if (!this._isEnabled) return;
        if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) return;

        for (const star of this._stars) {
            star.angle += star.angularSpeed * deltaSeconds;
            const x = Math.cos(star.angle) * star.radius;
            const z = Math.sin(star.angle) * star.radius;
            const y = this._baseHeight + star.heightOffset;
            star.mesh.position.set(x, y, z);
            star.mesh.rotation.y += star.spin * deltaSeconds;
            star.mesh.rotation.x += star.spin * 0.35 * deltaSeconds;
        }
    }

    dispose() {
        if (this.group && this.group.parent) {
            this.group.parent.remove(this.group);
        }
        for (const star of this._stars) {
            if (star.mesh) {
                if (star.material) star.material.dispose?.();
            }
        }
        this._stars.length = 0;
        if (this._sharedGeometry) {
            this._sharedGeometry.dispose?.();
            this._sharedGeometry = null;
        }
    }
}

export default OrbitingStars;
