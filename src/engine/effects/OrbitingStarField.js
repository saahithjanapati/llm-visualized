import * as THREE from 'three';
import BaseLayer from '../BaseLayer.js';

/**
 * Minimal ring of stars that orbit the GPT tower.  Implemented as a lightweight
 * BaseLayer so CoreEngine automatically ticks the effect every frame.
 */
export class OrbitingStarField extends BaseLayer {
    /**
     * @param {object} [opts]
     * @param {number} [opts.baseHeight]    – World-space Y value for the base of the tower.
     * @param {number} [opts.minHeightOffset] – Minimum Y offset above the base for a star.
     * @param {number} [opts.maxHeightOffset] – Maximum Y offset above the base for a star.
     * @param {number} [opts.minRadius]     – Minimum orbital radius from the tower centre.
     * @param {number} [opts.maxRadius]     – Maximum orbital radius from the tower centre.
     * @param {number} [opts.minSpeed]      – Minimum angular speed in radians/sec.
     * @param {number} [opts.maxSpeed]      – Maximum angular speed in radians/sec.
     * @param {number} [opts.count]         – Number of stars to spawn.
     * @param {number} [opts.size]          – Base size of each star mesh.
     * @param {number} [opts.twinkleAmount] – Scalar applied to the twinkle effect.
     */
    constructor(opts = {}) {
        super(-1);
        this.isActive = true; // allow CoreEngine to tick update()

        const defaults = {
            baseHeight: 0,
            minHeightOffset: 200,
            maxHeightOffset: 900,
            minRadius: 900,
            maxRadius: 1600,
            minSpeed: 0.08,
            maxSpeed: 0.22,
            count: 28,
            size: 26,
            twinkleAmount: 0.2,
            color: 0xffffff
        };
        this._opts = { ...defaults, ...opts };

        this._stars = [];
        this._enabled = true;
        this._elapsed = 0;

        this.root.position.y = this._opts.baseHeight || 0;
    }

    init(scene) {
        super.init(scene);

        const geometry = new THREE.IcosahedronGeometry(this._opts.size, 0);
        const material = new THREE.MeshBasicMaterial({
            color: new THREE.Color(this._opts.color),
            transparent: true,
            opacity: 0.85,
            depthWrite: false
        });

        for (let i = 0; i < this._opts.count; i++) {
            const mesh = new THREE.Mesh(geometry, material);
            mesh.renderOrder = -5; // keep subtle behind translucent components

            const radius = THREE.MathUtils.randFloat(this._opts.minRadius, this._opts.maxRadius);
            const angle = THREE.MathUtils.randFloat(0, Math.PI * 2);
            const speed = THREE.MathUtils.randFloat(this._opts.minSpeed, this._opts.maxSpeed) * (Math.random() < 0.5 ? -1 : 1);
            const yOffset = THREE.MathUtils.randFloat(this._opts.minHeightOffset, this._opts.maxHeightOffset);
            const baseScale = THREE.MathUtils.randFloat(0.65, 1.0);
            const twinkleSpeed = THREE.MathUtils.randFloat(0.6, 1.4);
            const bobSpeed = THREE.MathUtils.randFloat(0.25, 0.55);
            const bobAmount = THREE.MathUtils.randFloat(6, 18);
            const phase = THREE.MathUtils.randFloat(0, Math.PI * 2);

            mesh.scale.setScalar(baseScale);
            mesh.position.set(
                Math.cos(angle) * radius,
                yOffset,
                Math.sin(angle) * radius
            );

            this.root.add(mesh);
            this._stars.push({
                mesh,
                radius,
                angle,
                speed,
                yOffset,
                baseScale,
                twinkleSpeed,
                bobSpeed,
                bobAmount,
                phase
            });
        }

        this.root.visible = this._enabled;
    }

    /** Enable or disable the star field. */
    setEnabled(enabled) {
        this._enabled = !!enabled;
        if (this.root) {
            this.root.visible = this._enabled;
        }
    }

    /** Returns whether the star field is currently enabled. */
    isEnabled() {
        return !!this._enabled;
    }

    /** Updates the base height used for the orbit origin. */
    setBaseHeight(height) {
        if (typeof height !== 'number' || Number.isNaN(height)) return;
        this.root.position.y = height;
    }

    update(dt) {
        if (!this._enabled || !this._stars.length) return;
        this._elapsed += dt;
        const time = this._elapsed;
        const twinkleAmount = this._opts.twinkleAmount;

        for (const star of this._stars) {
            star.angle += star.speed * dt;
            const x = Math.cos(star.angle) * star.radius;
            const z = Math.sin(star.angle) * star.radius;
            const bob = Math.sin(time * star.bobSpeed + star.phase) * star.bobAmount;
            const scalePulse = 1 + Math.sin(time * star.twinkleSpeed + star.phase) * twinkleAmount;

            star.mesh.position.set(x, star.yOffset + bob, z);
            const s = star.baseScale * scalePulse;
            star.mesh.scale.setScalar(s);
        }
    }
}

export default OrbitingStarField;
