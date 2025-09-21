import * as THREE from 'three';

function getNowMillis() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

function attachTimeUpdater(material, uniforms, timeScale = 1) {
    if (!material || !uniforms || !uniforms.uTime) return;
    let startTime = null;
    const previousHook = material.onBeforeRender;
    material.onBeforeRender = function onBeforeRenderProxy(renderer, scene, camera, geometry, object, group) {
        if (typeof previousHook === 'function') {
            previousHook.call(this, renderer, scene, camera, geometry, object, group);
        }
        const now = getNowMillis();
        if (startTime === null) startTime = now;
        uniforms.uTime.value = ((now - startTime) / 1000) * timeScale;
    };
}

function withWorldPosition(shader) {
    if (!shader.vertexShader.includes('vWorldPosition')) {
        shader.vertexShader = shader.vertexShader.replace(
            '#include <common>',
            '#include <common>\nvarying vec3 vWorldPosition;'
        );
        shader.vertexShader = shader.vertexShader.replace(
            '#include <worldpos_vertex>',
            '#include <worldpos_vertex>\n    vWorldPosition = worldPosition.xyz;'
        );
    }
    if (!shader.fragmentShader.includes('vWorldPosition')) {
        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <common>',
            '#include <common>\nvarying vec3 vWorldPosition;'
        );
    }
}

export function createHolographicPanelMaterial(options = {}) {
    const {
        baseColor = 0x0a1a2f,
        emissiveColor = 0x1188ff,
        scanColor = 0x2affff,
        gridColor = 0x0cffd8,
        metalness = 0.85,
        roughness = 0.25,
        opacity = 0.9,
        fresnelPower = 3.25,
        edgeIntensity = 1.4,
        scanFrequency = 20.0,
        scanIntensity = 0.45,
        gridFrequency = 12.0,
        gridIntensity = 0.25,
        timeScale = 0.8
    } = options;

    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(baseColor),
        metalness,
        roughness,
        emissive: new THREE.Color(emissiveColor).multiplyScalar(0.15),
        emissiveIntensity: 0.6,
        transparent: true,
        opacity,
        side: THREE.FrontSide
    });

    material.customProgramCacheKey = () => 'SciFiPanelMaterial_v1';

    material.onBeforeCompile = function onBeforeCompile(shader) {
        withWorldPosition(shader);

        const uniforms = {
            uTime: { value: 0 },
            uScanFrequency: { value: scanFrequency },
            uScanIntensity: { value: scanIntensity },
            uGridFrequency: { value: gridFrequency },
            uGridIntensity: { value: gridIntensity },
            uEdgeIntensity: { value: edgeIntensity },
            uFresnelPower: { value: fresnelPower },
            uEdgeColor: { value: new THREE.Color(emissiveColor) },
            uScanColor: { value: new THREE.Color(scanColor) },
            uGridColor: { value: new THREE.Color(gridColor) }
        };

        Object.assign(shader.uniforms, uniforms);

        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <output_fragment>',
            `float fresnelTerm = pow(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), uFresnelPower);
float scanBand = sin(vWorldPosition.y * uScanFrequency + uTime * 2.2) * 0.5 + 0.5;
float gridFactor = sin(vWorldPosition.x * uGridFrequency + uTime * 1.7) * sin(vWorldPosition.z * uGridFrequency + uTime * 1.1);
gridFactor = smoothstep(0.0, 1.0, gridFactor * 0.5 + 0.5);
vec3 holographicGlow = uEdgeColor * fresnelTerm * uEdgeIntensity;
holographicGlow += uScanColor * scanBand * uScanIntensity;
holographicGlow += uGridColor * gridFactor * uGridIntensity;
vec3 finalLight = outgoingLight + holographicGlow;
gl_FragColor = vec4(finalLight, diffuseColor.a);
`);

        attachTimeUpdater(this, uniforms, timeScale);
    };

    return material;
}

export function createEnergyRingMaterial(options = {}) {
    const {
        baseColor = 0x0d2e3f,
        emissiveColor = 0x00c2ff,
        innerGlowColor = 0x7c4dff,
        metalness = 0.75,
        roughness = 0.2,
        opacity = 0.92,
        fresnelPower = 2.6,
        pulseFrequency = 1.6,
        pulseIntensity = 0.5,
        rimIntensity = 1.3,
        timeScale = 0.6
    } = options;

    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(baseColor),
        metalness,
        roughness,
        emissive: new THREE.Color(emissiveColor).multiplyScalar(0.2),
        emissiveIntensity: 0.7,
        transparent: true,
        opacity,
        side: THREE.DoubleSide
    });

    material.customProgramCacheKey = () => 'SciFiEnergyRing_v1';

    material.onBeforeCompile = function onBeforeCompile(shader) {
        withWorldPosition(shader);

        const uniforms = {
            uTime: { value: 0 },
            uPulseFrequency: { value: pulseFrequency },
            uPulseIntensity: { value: pulseIntensity },
            uRimIntensity: { value: rimIntensity },
            uFresnelPower: { value: fresnelPower },
            uOuterGlow: { value: new THREE.Color(emissiveColor) },
            uInnerGlow: { value: new THREE.Color(innerGlowColor) }
        };

        Object.assign(shader.uniforms, uniforms);

        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <output_fragment>',
            `vec3 viewDir = normalize(vViewPosition);
float fresnelTerm = pow(1.0 - abs(dot(normalize(normal), viewDir)), uFresnelPower);
float radialPulse = sin(uTime * uPulseFrequency + length(vWorldPosition.xy) * 1.5) * 0.5 + 0.5;
vec3 rimGlow = uOuterGlow * fresnelTerm * uRimIntensity;
vec3 coreGlow = uInnerGlow * radialPulse * uPulseIntensity;
vec3 finalLight = outgoingLight + rimGlow + coreGlow;
gl_FragColor = vec4(finalLight, diffuseColor.a);
`);

        attachTimeUpdater(this, uniforms, timeScale);
    };

    return material;
}

export function createPrismBeamMaterial(options = {}) {
    const {
        prismHalfWidth = 0.5,
        metalness = 0.4,
        roughness = 0.35,
        opacity = 0.95,
        baseEmissive = 0x00d8ff,
        highlightEmissive = 0xff4dd8,
        scanFrequency = 14.0,
        pulseScale = 0.35,
        fresnelPower = 3.8,
        timeScale = 1.2
    } = options;

    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0xffffff),
        metalness,
        roughness,
        emissive: new THREE.Color(baseEmissive).multiplyScalar(0.15),
        emissiveIntensity: 0.8,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        vertexColors: true
    });

    material.customProgramCacheKey = () => 'SciFiPrismBeam_v1';

    material.onBeforeCompile = function onBeforeCompile(shader) {
        const uniforms = {
            uTime: { value: 0 },
            uScanFrequency: { value: scanFrequency },
            uPulseScale: { value: pulseScale },
            uFresnelPower: { value: fresnelPower },
            uBaseEmissive: { value: new THREE.Color(baseEmissive) },
            uHighlightEmissive: { value: new THREE.Color(highlightEmissive) },
            prismHalfWidth: { value: prismHalfWidth }
        };

        Object.assign(shader.uniforms, uniforms);

        shader.vertexShader = shader.vertexShader.replace(
            '#include <common>',
            `#include <common>
varying vec3 vWorldPosition;
attribute vec3 colorStart;
attribute vec3 colorEnd;
varying vec3 vColorStart;
varying vec3 vColorEnd;
varying float vGradientT;
uniform float prismHalfWidth;
`
        );

        shader.vertexShader = shader.vertexShader.replace(
            '#include <worldpos_vertex>',
            '#include <worldpos_vertex>\n    vWorldPosition = worldPosition.xyz;'
        );

        shader.vertexShader = shader.vertexShader.replace(
            '#include <begin_vertex>',
            `#include <begin_vertex>
    vColorStart = colorStart;
    vColorEnd = colorEnd;
    vGradientT = clamp((position.x + prismHalfWidth) / (2.0 * prismHalfWidth), 0.0, 1.0);
`
        );

        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <common>',
            '#include <common>\nvarying vec3 vWorldPosition;\nvarying vec3 vColorStart;\nvarying vec3 vColorEnd;\nvarying float vGradientT;'
        );

        shader.fragmentShader = shader.fragmentShader.replace(
            'vec4 diffuseColor = vec4( diffuse, opacity );',
            `vec3 gradColor = mix(vColorStart, vColorEnd, vGradientT);
vec4 diffuseColor = vec4(gradColor, opacity);
`
        );

        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <output_fragment>',
            `vec3 viewDir = normalize(vViewPosition);
float fresnelTerm = pow(1.0 - abs(dot(normalize(normal), viewDir)), uFresnelPower);
float scanWave = sin(vWorldPosition.y * uScanFrequency + uTime * 5.0) * 0.5 + 0.5;
vec3 glow = uBaseEmissive * (scanWave * uPulseScale);
glow += uHighlightEmissive * fresnelTerm * 0.9;
vec3 finalLight = outgoingLight + glow;
gl_FragColor = vec4(finalLight, diffuseColor.a);
`
        );

        attachTimeUpdater(this, uniforms, timeScale);
    };

    return material;
}
