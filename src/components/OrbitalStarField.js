import * as THREE from 'three';

/**
 * Minimal collection of small stars orbiting around the GPT tower.
 * Stars are simple MeshBasic spheres with per-instance orbital speeds.
 */
export class OrbitalStarField {
    /**
     * @param {object} [options]
     * @param {number} [options.count=18]              – Number of star meshes to spawn.
     * @param {[number, number]} [options.radiusRange] – Min/Max orbital radius (world units).
     * @param {number} [options.baseHeight=0]          – Y position representing the base of the tower.
     * @param {number} [options.heightOffset=180]      – Base offset above the tower base where orbits sit.
     * @param {number} [options.verticalJitter=80]     – Random vertical jitter added on top of the offset.
     * @param {[number, number]} [options.speedRange]  – Min/Max angular speed in radians per second.
     * @param {[number, number]} [options.scaleRange]  – Min/Max scale multiplier for each star.
     * @param {number} [options.opacity=0.9]           – Star material opacity.
     * @param {THREE.ColorRepresentation} [options.color=0xffffff] – Base star colour.
     */
    constructor(options = {}) {
        const {
            count = 18,
            radiusRange = [650, 1100],
            baseHeight = 0,
            heightOffset = 180,
            verticalJitter = 80,
            speedRange = [0.25, 0.55],
            scaleRange = [0.6, 1.4],
            opacity = 0.9,
            color = 0xffffff
        } = options;

        this.group = new THREE.Group();
        this.group.position.y = baseHeight;

        this._enabled = true;
        this._stars = [];
        this._sharedGeometry = new THREE.IcosahedronGeometry(6, 0);
        const baseColor = new THREE.Color(color);

        for (let i = 0; i < count; i++) {
            const material = new THREE.MeshBasicMaterial({
                color: baseColor.clone().offsetHSL(0, (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1),
                transparent: true,
                opacity,
                depthWrite: false
            });

            const mesh = new THREE.Mesh(this._sharedGeometry, material);
            const radius = THREE.MathUtils.lerp(radiusRange[0], radiusRange[1], Math.random());
            const angle = Math.random() * Math.PI * 2;
            const speed = THREE.MathUtils.lerp(speedRange[0], speedRange[1], Math.random());
            const spin = THREE.MathUtils.lerp(0.2, 0.8, Math.random());
            const vertical = heightOffset + (Math.random() - 0.5) * verticalJitter;
            const scale = THREE.MathUtils.lerp(scaleRange[0], scaleRange[1], Math.random());
            mesh.scale.setScalar(scale);

            mesh.position.set(
                Math.cos(angle) * radius,
                vertical,
                Math.sin(angle) * radius
            );

            this.group.add(mesh);
            this._stars.push({ mesh, radius, angle, speed, spin, vertical });
        }
    }

    /** Update orbital rotation. */
    update(dt = 0) {
        if (!this._enabled || !Number.isFinite(dt) || dt <= 0) return;
        for (const star of this._stars) {
            star.angle += star.speed * dt;
            const mesh = star.mesh;
            mesh.position.x = Math.cos(star.angle) * star.radius;
            mesh.position.z = Math.sin(star.angle) * star.radius;
            mesh.position.y = star.vertical;
            mesh.rotation.y += star.spin * dt;
        }
    }

    /** Enable/disable star visibility and animation. */
    setEnabled(enabled) {
        this._enabled = !!enabled;
        this.group.visible = this._enabled;
    }

    /** Dispose of geometries and materials. */
    dispose() {
        for (const star of this._stars) {
            if (Array.isArray(star.mesh.material)) {
                star.mesh.material.forEach(mat => mat?.dispose?.());
            } else {
                star.mesh.material?.dispose?.();
            }
        }
        this._stars.length = 0;
        this._sharedGeometry?.dispose?.();
    }
}

export default OrbitalStarField;
