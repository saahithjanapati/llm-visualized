import * as THREE from 'three';

const TMP_COLOR = new THREE.Color();

function sampleRadius(minRadius, maxRadius, exponent = 0.5) {
    const t = Math.random();
    const eased = Math.pow(t, exponent);
    return THREE.MathUtils.lerp(minRadius, maxRadius, eased);
}

function clampRange(minValue, maxValue) {
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
        return [0, 0];
    }
    if (maxValue <= minValue) {
        return [minValue, minValue];
    }
    return [minValue, maxValue];
}

function createPointsLayer(config) {
    const {
        count,
        baseY,
        minHeight,
        maxHeight,
        minRadius,
        maxRadius,
        size,
        colorVariation,
        opacity,
        rotationSpeedMultiplier
    } = config;

    const [minH, maxH] = clampRange(minHeight, maxHeight);
    const [minR, maxR] = clampRange(minRadius, maxRadius);

    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = sampleRadius(minR, maxR, 0.35);
        const heightT = Math.random();
        const y = baseY + minH + heightT * (maxH - minH);

        const idx = i * 3;
        positions[idx] = Math.cos(angle) * radius;
        positions[idx + 1] = y;
        positions[idx + 2] = Math.sin(angle) * radius;

        const hueOffset = (Math.random() - 0.5) * colorVariation.hue;
        const satOffset = (Math.random() - 0.5) * colorVariation.sat;
        const lightOffset = (Math.random() - 0.5) * colorVariation.light;
        TMP_COLOR.setHSL(
            THREE.MathUtils.clamp(colorVariation.baseHue + hueOffset, 0, 1),
            THREE.MathUtils.clamp(colorVariation.baseSat + satOffset, 0, 1),
            THREE.MathUtils.clamp(colorVariation.baseLight + lightOffset, 0, 1)
        );
        colors[idx] = TMP_COLOR.r;
        colors[idx + 1] = TMP_COLOR.g;
        colors[idx + 2] = TMP_COLOR.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size,
        sizeAttenuation: true,
        vertexColors: true,
        transparent: true,
        opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });

    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;

    const group = new THREE.Group();
    group.add(points);

    return {
        group,
        geometry,
        material,
        speedMultiplier: rotationSpeedMultiplier
    };
}

export class SkyStarField {
    constructor(options = {}) {
        const {
            baseY = 0,
            minHeight = 600,
            maxHeight = 20000,
            minRadius = 800,
            maxRadius = 9000,
            starCount = 1400,
            rotationSpeed = 0.08
        } = options;

        this.baseY = Number.isFinite(baseY) ? baseY : 0;
        this._rotationSpeed = Number.isFinite(rotationSpeed) ? rotationSpeed : 0.08;
        this._layers = [];
        this.group = new THREE.Group();
        this.group.name = 'SkyStarField';
        this.group.matrixAutoUpdate = true;

        const layerConfigs = [
            {
                count: Math.floor(starCount * 0.65),
                baseY: this.baseY,
                minHeight,
                maxHeight,
                minRadius,
                maxRadius: maxRadius * 0.85,
                size: 90,
                opacity: 0.85,
                rotationSpeedMultiplier: 1.0,
                colorVariation: {
                    baseHue: 0.6,
                    baseSat: 0.35,
                    baseLight: 0.85,
                    hue: 0.08,
                    sat: 0.15,
                    light: 0.1
                }
            },
            {
                count: Math.max(0, starCount - Math.floor(starCount * 0.65)),
                baseY: this.baseY,
                minHeight: minHeight + 1500,
                maxHeight: maxHeight + 6000,
                minRadius: minRadius * 0.4,
                maxRadius: maxRadius * 1.2,
                size: 120,
                opacity: 0.7,
                rotationSpeedMultiplier: -0.45,
                colorVariation: {
                    baseHue: 0.58,
                    baseSat: 0.28,
                    baseLight: 0.92,
                    hue: 0.06,
                    sat: 0.12,
                    light: 0.08
                }
            }
        ];

        layerConfigs.forEach(cfg => {
            if (!cfg.count) return;
            const layer = createPointsLayer(cfg);
            this.group.add(layer.group);
            this._layers.push(layer);
        });
    }

    update(dt) {
        if (!this._layers || !this._layers.length) return;
        const delta = Number.isFinite(dt) ? dt : 0;
        this._layers.forEach(layer => {
            if (!layer || !layer.group) return;
            const speed = this._rotationSpeed * (layer.speedMultiplier || 1);
            layer.group.rotation.y += speed * delta;
        });
    }

    dispose() {
        if (!this._layers) return;
        this._layers.forEach(layer => {
            if (!layer) return;
            if (layer.group && layer.group.parent) {
                layer.group.parent.remove(layer.group);
            }
            if (layer.geometry) layer.geometry.dispose();
            if (layer.material) layer.material.dispose();
        });
        this._layers.length = 0;
    }
}

export default SkyStarField;
