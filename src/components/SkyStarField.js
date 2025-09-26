import * as THREE from 'three';

/**
 * Creates and manages a rotating star field that orbits around the GPT tower.
 * Stars are distributed in a cylindrical volume that begins above the tower
 * base and extends across the scene "sky" area. The group is added directly to
 * the pipeline's engine scene and registers a frame update so the field slowly
 * rotates around the vertical axis.
 */
export class SkyStarField {
    constructor(pipeline, options = {}) {
        if (!pipeline?.engine || !pipeline.engine.scene) {
            throw new Error('SkyStarField requires a pipeline with an active engine');
        }

        this.pipeline = pipeline;
        this.engine = pipeline.engine;
        this.rotationSpeed = typeof options.rotationSpeed === 'number' ? options.rotationSpeed : 0.015;
        this.starCount = Math.max(1, Math.floor(options.starCount ?? 900));
        this.minHeightOffset = typeof options.minHeightOffset === 'number' ? options.minHeightOffset : 250;
        this.verticalPadding = typeof options.verticalPadding === 'number' ? options.verticalPadding : 6000;
        this.innerRadiusPadding = typeof options.innerRadiusPadding === 'number' ? options.innerRadiusPadding : 1200;
        this.innerRadiusOverride = typeof options.innerRadius === 'number' ? options.innerRadius : null;
        this.outerRadiusOverride = typeof options.outerRadius === 'number' ? options.outerRadius : null;
        this.group = new THREE.Group();
        this.group.name = 'SkyStarField';

        const layout = this._computeLayout();
        this.bounds = layout;

        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.starCount * 3);
        const colors = new Float32Array(this.starCount * 3);
        const color = new THREE.Color();

        for (let i = 0; i < this.starCount; i += 1) {
            const angle = Math.random() * Math.PI * 2;
            const radius = this._randomRadius(layout.innerRadius, layout.outerRadius);
            const y = layout.minY + Math.random() * (layout.maxY - layout.minY);

            positions[i * 3 + 0] = Math.cos(angle) * radius;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = Math.sin(angle) * radius;

            const hue = 0.55 + Math.random() * 0.1;
            const saturation = 0.15 + Math.random() * 0.25;
            const lightness = 0.75 + Math.random() * 0.2;
            color.setHSL(hue, saturation, lightness);
            color.toArray(colors, i * 3);
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeBoundingSphere();

        const material = new THREE.PointsMaterial({
            size: options.size ?? 160,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.85,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            vertexColors: true
        });

        this.points = new THREE.Points(geometry, material);
        this.points.frustumCulled = false;
        this.group.add(this.points);
        this.engine.scene.add(this.group);

        this._baseOpacity = material.opacity;
        this._updateCallback = (dt) => {
            if (!this.group.visible) return;
            this.group.rotation.y += dt * this.rotationSpeed;
        };

        if (typeof this.engine.registerFrameUpdate === 'function') {
            this.engine.registerFrameUpdate(this._updateCallback);
        }
    }

    _computeLayout() {
        const layers = Array.isArray(this.pipeline?._layers) ? this.pipeline._layers : [];
        const tmpBox = new THREE.Box3();
        let baseMinY = 0;
        let topMaxY = baseMinY + 4000;
        let approxWidth = 2000;
        let approxDepth = 2000;

        if (layers.length > 0 && layers[0]?.root) {
            tmpBox.setFromObject(layers[0].root);
            if (Number.isFinite(tmpBox.min.y)) baseMinY = tmpBox.min.y;
            const width = Math.abs(tmpBox.max.x - tmpBox.min.x);
            const depth = Math.abs(tmpBox.max.z - tmpBox.min.z);
            if (Number.isFinite(width) && width > 0) approxWidth = width;
            if (Number.isFinite(depth) && depth > 0) approxDepth = depth;
        }

        if (layers.length > 0) {
            const lastLayer = layers[layers.length - 1];
            if (lastLayer?.root) {
                tmpBox.setFromObject(lastLayer.root);
                if (Number.isFinite(tmpBox.max.y)) topMaxY = tmpBox.max.y;
                const width = Math.abs(tmpBox.max.x - tmpBox.min.x);
                const depth = Math.abs(tmpBox.max.z - tmpBox.min.z);
                if (Number.isFinite(width) && width > approxWidth) approxWidth = width;
                if (Number.isFinite(depth) && depth > approxDepth) approxDepth = depth;
            }
        }

        if (!Number.isFinite(topMaxY) || topMaxY <= baseMinY) {
            topMaxY = baseMinY + 4000;
        }

        const rawMinY = baseMinY + this.minHeightOffset;
        let rawMaxY = topMaxY + this.verticalPadding;
        if (!Number.isFinite(rawMaxY) || rawMaxY <= rawMinY) {
            rawMaxY = rawMinY + 2000;
        }

        const baseRadius = Math.max(approxWidth, approxDepth) * 0.5;
        const innerRadius = this.innerRadiusOverride ?? (baseRadius + this.innerRadiusPadding);
        let outerRadius = this.outerRadiusOverride ?? (innerRadius + Math.max(3000, (topMaxY - baseMinY) * 0.35));
        if (!Number.isFinite(outerRadius) || outerRadius <= innerRadius) {
            outerRadius = innerRadius + 4000;
        }

        return {
            minY: rawMinY,
            maxY: rawMaxY,
            innerRadius,
            outerRadius
        };
    }

    _randomRadius(inner, outer) {
        const innerSq = inner * inner;
        const outerSq = outer * outer;
        const t = Math.random();
        return Math.sqrt(innerSq + (outerSq - innerSq) * t);
    }

    setEnabled(enabled) {
        const visible = !!enabled;
        this.group.visible = visible;
        if (this.points?.material) {
            this.points.material.opacity = visible ? this._baseOpacity : 0;
            this.points.material.needsUpdate = true;
        }
    }

    dispose() {
        if (this.points) {
            this.points.geometry?.dispose?.();
            this.points.material?.dispose?.();
        }
        if (this.group && this.engine?.scene) {
            this.engine.scene.remove(this.group);
        }
        if (this._updateCallback && typeof this.engine?.unregisterFrameUpdate === 'function') {
            this.engine.unregisterFrameUpdate(this._updateCallback);
        }
        this.points = null;
    }
}

export default SkyStarField;
