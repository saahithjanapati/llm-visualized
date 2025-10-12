import * as THREE from 'three';
import { CAPTION_TEXT_Y_POS } from '../utils/constants.js';

const DEFAULTS = {
    orbitRadius: 16000,
    planetRadius: 3200,
    planetHeight: 6000,
    orbitSpeed: 0.05,
    planetSpinSpeed: 0.12,
    starRotationSpeed: 0.01,
    starCount: 2200,
    starInnerRadius: 18000,
    starOuterRadius: 52000,
    starSize: 160,
    gridSize: 48000,
    gridRepeat: 60,
    gridLineWidth: 0.028,
    gridDotSize: 0.16,
    gridOpacity: 0.8,
    gridFadeStart: 6000,
    gridFadeEnd: 26000,
    gridYOffset: 300,
    gridColor: 0x14d9ff,
    gridBackground: 0x02060f
};

function randomDirection(target = new THREE.Vector3()) {
    const u = Math.random();
    const v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    const sinPhi = Math.sin(phi);
    target.set(
        Math.cos(theta) * sinPhi,
        Math.cos(phi),
        Math.sin(theta) * sinPhi
    );
    return target;
}

function createGridMaterial(options) {
    const uniforms = {
        uGridColor:     { value: new THREE.Color(options.gridColor) },
        uBackground:    { value: new THREE.Color(options.gridBackground) },
        uOpacity:       { value: options.gridOpacity },
        uRepeat:        { value: new THREE.Vector2(options.gridRepeat, options.gridRepeat) },
        uFadeStart:     { value: options.gridFadeStart },
        uFadeEnd:       { value: options.gridFadeEnd },
        uLineWidth:     { value: options.gridLineWidth },
        uDotSize:       { value: options.gridDotSize }
    };

    return new THREE.ShaderMaterial({
        uniforms,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        vertexShader: `
            varying vec2 vUv;
            varying float vRadius;
            uniform vec2 uRepeat;
            void main() {
                vUv = uv * uRepeat;
                vec4 worldPos = modelMatrix * vec4(position, 1.0);
                vRadius = length(worldPos.xz);
                gl_Position = projectionMatrix * viewMatrix * worldPos;
            }
        `,
        fragmentShader: `
            varying vec2 vUv;
            varying float vRadius;
            uniform vec3 uGridColor;
            uniform vec3 uBackground;
            uniform float uOpacity;
            uniform float uFadeStart;
            uniform float uFadeEnd;
            uniform float uLineWidth;
            uniform float uDotSize;

            float lineIntensity(vec2 gv, float width) {
                float line = min(gv.x, gv.y);
                return 1.0 - smoothstep(width, width + 0.01, line);
            }

            void main() {
                vec2 gv = abs(fract(vUv) - 0.5);
                float line = lineIntensity(gv, uLineWidth);
                float dots = 1.0 - smoothstep(uDotSize, uDotSize + 0.02, length(gv));
                float intensity = max(line, dots * 0.45);
                float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, vRadius);
                float alpha = uOpacity * intensity * fade;
                if (alpha <= 0.001) discard;
                vec3 color = mix(uBackground, uGridColor, intensity);
                gl_FragColor = vec4(color, alpha);
            }
        `
    });
}

export class SciFiEnvironment {
    constructor(options = {}) {
        this.options = { ...DEFAULTS, ...options };
        this.group = new THREE.Group();
        this.group.name = 'SciFiEnvironment';

        this._planetPivot = new THREE.Group();
        this._planetPivot.position.set(0, this.options.planetHeight, 0);
        this._planetPivot.name = 'SciFiPlanetPivot';
        this.group.add(this._planetPivot);

        this._planet = this._createPlanet();
        this._planetPivot.add(this._planet);

        this._planetGlow = this._createPlanetGlow();
        if (this._planetGlow) {
            this._planetGlow.position.copy(this._planet.position);
            this._planetPivot.add(this._planetGlow);
        }

        this._stars = this._createStars();
        this.group.add(this._stars);

        this._grid = this._createGrid();
        this.group.add(this._grid);

        this._orbitAngle = 0;
        this._planetSpin = 0;
    }

    addToScene(scene) {
        if (scene && !this.group.parent) {
            scene.add(this.group);
        }
    }

    removeFromScene() {
        if (this.group.parent) {
            this.group.parent.remove(this.group);
        }
    }

    update(dt = 0) {
        if (!Number.isFinite(dt) || dt <= 0) return;
        this._orbitAngle += dt * this.options.orbitSpeed;
        this._planetSpin += dt * this.options.planetSpinSpeed;
        if (this._planetPivot) {
            this._planetPivot.rotation.y = this._orbitAngle;
        }
        if (this._planet) {
            this._planet.rotation.y = this._planetSpin;
        }
        if (this._planetGlow) {
            this._planetGlow.rotation.y = this._planetSpin * 0.5;
        }
        if (this._stars) {
            this._stars.rotation.y += dt * this.options.starRotationSpeed;
        }
    }

    dispose() {
        this.removeFromScene();
        if (this._stars) {
            this._stars.geometry?.dispose();
            this._stars.material?.dispose();
            this._stars = null;
        }
        if (this._planet) {
            this._planet.geometry?.dispose();
            this._planet.material?.dispose();
            this._planet = null;
        }
        if (this._planetGlow) {
            this._planetGlow.geometry?.dispose();
            this._planetGlow.material?.dispose();
            this._planetGlow = null;
        }
        if (this._grid) {
            this._grid.geometry?.dispose();
            this._grid.material?.dispose();
            this._grid = null;
        }
    }

    _createPlanet() {
        const geometry = new THREE.SphereGeometry(this.options.planetRadius, 64, 64);
        const material = new THREE.MeshStandardMaterial({
            color: 0x2046ff,
            emissive: 0x1230a5,
            emissiveIntensity: 1.35,
            metalness: 0.25,
            roughness: 0.4
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(0, 0, -this.options.orbitRadius);
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        return mesh;
    }

    _createPlanetGlow() {
        const geometry = new THREE.SphereGeometry(this.options.planetRadius * 1.08, 48, 48);
        const material = new THREE.MeshBasicMaterial({
            color: 0x4be1ff,
            transparent: true,
            opacity: 0.18,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(0, 0, -this.options.orbitRadius);
        return mesh;
    }

    _createStars() {
        const geometry = new THREE.BufferGeometry();
        const count = this.options.starCount;
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);
        const color = new THREE.Color();
        const dir = new THREE.Vector3();
        for (let i = 0; i < count; i++) {
            randomDirection(dir);
            const radius = THREE.MathUtils.lerp(this.options.starInnerRadius, this.options.starOuterRadius, Math.random());
            dir.multiplyScalar(radius);
            positions[i * 3 + 0] = dir.x;
            positions[i * 3 + 1] = dir.y;
            positions[i * 3 + 2] = dir.z;

            const hue = 0.55 + Math.random() * 0.08;
            const sat = 0.45 + Math.random() * 0.3;
            const light = 0.5 + Math.random() * 0.3;
            color.setHSL(hue, sat, light);
            colors[i * 3 + 0] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: this.options.starSize,
            sizeAttenuation: true,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            vertexColors: true,
            opacity: 0.8
        });

        const points = new THREE.Points(geometry, material);
        points.name = 'SciFiStarField';
        return points;
    }

    _createGrid() {
        const geometry = new THREE.PlaneGeometry(1, 1, 1, 1);
        const material = createGridMaterial(this.options);
        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        mesh.scale.set(this.options.gridSize, this.options.gridSize, 1);
        const y = (typeof this.options.gridY === 'number')
            ? this.options.gridY
            : CAPTION_TEXT_Y_POS + this.options.gridYOffset;
        mesh.position.set(0, y, 0);
        mesh.name = 'SciFiGrid';
        mesh.frustumCulled = false;
        mesh.renderOrder = -10;
        return mesh;
    }
}

export default SciFiEnvironment;
