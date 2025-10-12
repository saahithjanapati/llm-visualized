import * as THREE from 'three';

/**
 * Decorative sci-fi environment that can be toggled on/off. Consists of a
 * rotating planet, a twinkling starfield and a holographic floor grid that sits
 * beneath the GPT-2 tower.
 */
export class SciFiEnvironment {
    constructor() {
        this.group = new THREE.Group();
        this.group.name = 'SciFiEnvironment';
        this.group.visible = false;

        this._time = 0;

        this._planet = this._createPlanet();
        this.group.add(this._planet);

        this._starfield = this._createStarfield();
        this.group.add(this._starfield);

        this._grid = this._createGrid();
        this.group.add(this._grid);
    }

    setEnabled(enabled) {
        this.group.visible = !!enabled;
    }

    update(dt = 0) {
        if (!this.group.visible) return;

        this._time += dt;
        if (this._planet) {
            this._planet.rotation.y += dt * 0.08;
            this._planet.rotation.z = Math.sin(this._time * 0.15) * 0.1;
        }
        if (this._starfield?.material?.uniforms) {
            this._starfield.material.uniforms.uTime.value = this._time * 0.6;
        }
        if (this._grid?.material?.uniforms) {
            this._grid.material.uniforms.uTime.value = this._time;
        }
    }

    dispose() {
        this.group.traverse((obj) => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (obj.material.uniformsTexture) {
                    Object.values(obj.material.uniformsTexture).forEach((tex) => tex?.dispose?.());
                }
                if (Array.isArray(obj.material)) {
                    obj.material.forEach((m) => m.dispose?.());
                } else if (obj.material.dispose) {
                    obj.material.dispose();
                }
            }
        });
    }

    _createPlanet() {
        const geometry = new THREE.SphereGeometry(5200, 64, 64);
        const texture = this._generatePlanetTexture(2048);
        const material = new THREE.MeshStandardMaterial({
            map: texture,
            metalness: 0.2,
            roughness: 0.7,
            emissive: new THREE.Color(0x0b1d3b),
            emissiveIntensity: 0.35
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(-9000, 2500, -16000);
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        return mesh;
    }

    _generatePlanetTexture(size) {
        const canvas = document?.createElement?.('canvas');
        if (!canvas) return null;
        canvas.width = canvas.height = size;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;

        const gradient = ctx.createRadialGradient(
            size * 0.35, size * 0.3, size * 0.05,
            size * 0.5, size * 0.5, size * 0.6
        );
        gradient.addColorStop(0, '#1d6db2');
        gradient.addColorStop(0.4, '#10325b');
        gradient.addColorStop(1, '#050914');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, size, size);

        ctx.globalAlpha = 0.14;
        ctx.fillStyle = '#0ff9ff';
        const noiseCount = size * 6;
        for (let i = 0; i < noiseCount; i++) {
            const r = Math.random() * 2 + 0.5;
            ctx.beginPath();
            ctx.arc(Math.random() * size, Math.random() * size, r, 0, Math.PI * 2);
            ctx.fill();
        }

        const canvasTexture = new THREE.CanvasTexture(canvas);
        canvasTexture.wrapS = THREE.MirroredRepeatWrapping;
        canvasTexture.wrapT = THREE.MirroredRepeatWrapping;
        canvasTexture.anisotropy = 4;
        return canvasTexture;
    }

    _createStarfield() {
        const starCount = 1500;
        const radius = 32000;
        const positions = new Float32Array(starCount * 3);
        const sizes = new Float32Array(starCount);
        const phases = new Float32Array(starCount);

        for (let i = 0; i < starCount; i++) {
            const phi = Math.acos(2 * Math.random() - 1);
            const theta = Math.random() * Math.PI * 2;
            const r = radius * (0.35 + Math.random() * 0.65);
            const x = Math.sin(phi) * Math.cos(theta) * r;
            const y = Math.cos(phi) * r * 0.35;
            const z = Math.sin(phi) * Math.sin(theta) * r;
            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;
            sizes[i] = 18 + Math.random() * 18;
            phases[i] = Math.random() * Math.PI * 2;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
        geometry.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));

        const material = new THREE.ShaderMaterial({
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
            uniforms: {
                uTime: { value: 0 },
                uColor: { value: new THREE.Color(0x67d4ff) },
                uBaseOpacity: { value: 0.8 }
            },
            vertexShader: /* glsl */`
                attribute float aSize;
                attribute float aPhase;
                varying float vPhase;
                void main() {
                    vPhase = aPhase;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    float size = aSize * (300.0 / max(1.0, -mvPosition.z));
                    gl_PointSize = clamp(size, 2.0, 60.0);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: /* glsl */`
                uniform vec3 uColor;
                uniform float uTime;
                uniform float uBaseOpacity;
                varying float vPhase;
                void main() {
                    vec2 uv = gl_PointCoord - 0.5;
                    float dist = length(uv);
                    float mask = smoothstep(0.5, 0.0, dist);
                    float twinkle = 0.65 + 0.35 * sin(uTime + vPhase);
                    gl_FragColor = vec4(uColor, mask * uBaseOpacity * twinkle);
                }
            `
        });

        const points = new THREE.Points(geometry, material);
        points.frustumCulled = false;
        return points;
    }

    _createGrid() {
        const size = 64000;
        const geometry = new THREE.PlaneGeometry(size, size, 1, 1);
        const material = new THREE.ShaderMaterial({
            transparent: true,
            depthWrite: true,
            uniforms: {
                uTime: { value: 0 },
                uSpacing: { value: 2200 },
                uLineWidth: { value: 0.02 },
                uLineColor: { value: new THREE.Color(0x28d3ff) },
                uBackgroundColor: { value: new THREE.Color(0x05060b) },
                uGlowStrength: { value: 0.35 }
            },
            vertexShader: /* glsl */`
                varying vec2 vWorldPos;
                void main() {
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPos = worldPosition.xz;
                    gl_Position = projectionMatrix * viewMatrix * worldPosition;
                }
            `,
            fragmentShader: /* glsl */`
                varying vec2 vWorldPos;
                uniform float uSpacing;
                uniform float uLineWidth;
                uniform vec3 uLineColor;
                uniform vec3 uBackgroundColor;
                uniform float uGlowStrength;
                uniform float uTime;

                float gridMask(float coord) {
                    float local = abs(fract(coord) - 0.5);
                    return smoothstep(uLineWidth, 0.0, local);
                }

                void main() {
                    vec2 coord = vWorldPos / uSpacing;
                    float lineX = gridMask(coord.x);
                    float lineZ = gridMask(coord.y);
                    float line = max(lineX, lineZ);

                    vec2 majorCoord = coord / 5.0;
                    float major = max(gridMask(majorCoord.x), gridMask(majorCoord.y));
                    float glow = pow(line, 1.2) * uGlowStrength + pow(major, 1.5) * (uGlowStrength + 0.15);
                    float pulse = 0.85 + 0.15 * sin(uTime * 0.5);

                    vec3 color = mix(uBackgroundColor, uLineColor, line * 0.8 + major * 0.2);
                    color += uLineColor * glow * pulse;
                    float alpha = 0.65 + 0.3 * line;
                    gl_FragColor = vec4(color, clamp(alpha, 0.0, 1.0));
                }
            `,
            side: THREE.DoubleSide
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(0, -1800, 0);
        mesh.receiveShadow = false;
        mesh.castShadow = false;
        mesh.renderOrder = -5;
        return mesh;
    }
}

export default SciFiEnvironment;
