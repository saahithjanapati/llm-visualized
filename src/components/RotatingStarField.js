import * as THREE from 'three';

/**
 * Simple rotating particle field that simulates a band of stars orbiting
 * around the GPT tower. Stars are generated within a cylindrical volume
 * above the provided base height and rotate around the Y axis.
 */
export class RotatingStarField {
    /**
     * @param {THREE.Scene} scene - Scene to attach the stars to.
     * @param {object} [options]
     * @param {THREE.Vector3} [options.center] - World-space centre of rotation.
     * @param {number} [options.baseY] - Base world-space Y used for local offsets.
     * @param {number} [options.minRadius] - Inner radius of the cylindrical band.
     * @param {number} [options.maxRadius] - Outer radius of the cylindrical band.
     * @param {number} [options.minY] - Minimum world-space Y for star placement.
     * @param {number} [options.maxY] - Maximum world-space Y for star placement.
     * @param {number} [options.starCount] - Number of stars to spawn.
     * @param {number} [options.rotationSpeed] - Radians per second.
     * @param {number} [options.starSize] - Base rendered size for each star.
     */
    constructor(scene, options = {}) {
        if (!scene || typeof scene.add !== 'function') {
            throw new Error('RotatingStarField requires a valid THREE.Scene');
        }

        this.scene = scene;
        this.rotationSpeed = Number.isFinite(options.rotationSpeed)
            ? options.rotationSpeed
            : 0.12;

        const centre = (options.center instanceof THREE.Vector3)
            ? options.center.clone()
            : new THREE.Vector3();
        const baseY = Number.isFinite(options.baseY) ? options.baseY : centre.y;

        this.group = new THREE.Group();
        this.group.name = 'RotatingStarField';
        this.group.position.set(centre.x, baseY, centre.z);

        const minRadius = Math.max(0, Number(options.minRadius) || 0);
        let maxRadius = Number(options.maxRadius);
        if (!Number.isFinite(maxRadius) || maxRadius <= minRadius) {
            maxRadius = minRadius + 1000;
        }

        const minY = Number(options.minY);
        const maxY = Number(options.maxY);
        this._minY = Number.isFinite(minY) ? minY : baseY + 500;
        const fallbackMaxY = this._minY + 3000;
        this._maxY = Number.isFinite(maxY) && maxY > this._minY ? maxY : fallbackMaxY;

        const starCount = Math.max(1, Math.floor(Number(options.starCount) || 600));
        const starSize = Number.isFinite(options.starSize) ? options.starSize : 160;

        const positions = new Float32Array(starCount * 3);
        const colors = new Float32Array(starCount * 3);
        const tmpColor = new THREE.Color();

        for (let i = 0; i < starCount; i++) {
            const angle = Math.random() * Math.PI * 2;
            const radius = minRadius + Math.random() * (maxRadius - minRadius);
            const yWorld = this._minY + Math.random() * (this._maxY - this._minY);

            positions[i * 3 + 0] = Math.cos(angle) * radius;
            positions[i * 3 + 1] = yWorld - baseY;
            positions[i * 3 + 2] = Math.sin(angle) * radius;

            const hue = 0.55 + Math.random() * 0.1; // cool blue-white hues
            const saturation = 0.15 + Math.random() * 0.25;
            const lightness = 0.75 + Math.random() * 0.2;
            tmpColor.setHSL(hue, saturation, lightness);
            colors[i * 3 + 0] = tmpColor.r;
            colors[i * 3 + 1] = tmpColor.g;
            colors[i * 3 + 2] = tmpColor.b;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: starSize,
            sizeAttenuation: false,
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        this.points = new THREE.Points(geometry, material);
        this.points.frustumCulled = false;
        this.group.add(this.points);
        this.scene.add(this.group);
        this.setEnabled(true);
    }

    update(dt = 0) {
        if (!this.group) return;
        const delta = Number.isFinite(dt) ? dt : 0;
        if (delta === 0) return;
        this.group.rotation.y += delta * this.rotationSpeed;
    }

    setEnabled(enabled) {
        const visible = !!enabled;
        if (this.group) {
            this.group.visible = visible;
        }
        this._visible = visible;
    }

    dispose() {
        if (this.scene && this.group && this.group.parent === this.scene) {
            this.scene.remove(this.group);
        }
        if (this.points) {
            if (this.points.geometry) this.points.geometry.dispose();
            if (this.points.material) this.points.material.dispose();
        }
        this.points = null;
        this.group = null;
        this.scene = null;
    }
}

export default RotatingStarField;
