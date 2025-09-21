import * as THREE from 'three';

const textureCache = new Map();

function createCanvas(size = 512) {
    if (typeof document !== 'undefined' && document.createElement) {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = size;
        return canvas;
    }
    if (typeof OffscreenCanvas !== 'undefined') {
        return new OffscreenCanvas(size, size);
    }
    return null;
}

function registerTexture(key, generator) {
    if (textureCache.has(key)) {
        return textureCache.get(key);
    }
    const canvas = generator();
    if (!canvas) {
        textureCache.set(key, null);
        return null;
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = 8;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    textureCache.set(key, texture);
    return texture;
}

function drawNeonGridBackground(size = 512) {
    const canvas = createCanvas(size, true);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const gradient = ctx.createLinearGradient(0, 0, size, size);
    gradient.addColorStop(0, '#0b1933');
    gradient.addColorStop(0.45, '#040b16');
    gradient.addColorStop(1, '#02060c');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    ctx.globalCompositeOperation = 'lighter';
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.12)';
    ctx.lineWidth = 2;

    const spacing = size / 10;
    for (let i = -size; i <= size * 2; i += spacing) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i - size, size);
        ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(44, 130, 255, 0.15)';
    for (let i = -size; i <= size * 2; i += spacing * 0.5) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(size, i - size);
        ctx.stroke();
    }

    ctx.globalCompositeOperation = 'source-over';
    const radial = ctx.createRadialGradient(size * 0.5, size * 0.5, size * 0.05, size * 0.5, size * 0.5, size * 0.5);
    radial.addColorStop(0, 'rgba(0, 180, 255, 0.35)');
    radial.addColorStop(0.45, 'rgba(0, 80, 160, 0.15)');
    radial.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = radial;
    ctx.fillRect(0, 0, size, size);

    return canvas;
}

function drawEmissiveScanTexture(size = 256) {
    const canvas = createCanvas(size, true);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.clearRect(0, 0, size, size);

    const gradient = ctx.createLinearGradient(0, 0, 0, size);
    gradient.addColorStop(0, 'rgba(0, 255, 255, 0.0)');
    gradient.addColorStop(0.25, 'rgba(0, 255, 255, 0.5)');
    gradient.addColorStop(0.5, 'rgba(0, 255, 255, 0.9)');
    gradient.addColorStop(0.75, 'rgba(0, 255, 255, 0.5)');
    gradient.addColorStop(1, 'rgba(0, 255, 255, 0.0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = 'rgba(0, 255, 180, 0.65)';
    const lineCount = 24;
    const lineHeight = size / (lineCount * 2);
    for (let i = 0; i < lineCount; i++) {
        const y = (i / lineCount) * size;
        ctx.fillRect(0, y, size, lineHeight);
    }
    ctx.globalCompositeOperation = 'source-over';

    return canvas;
}

function drawNoiseTexture(size = 256) {
    const canvas = createCanvas(size, true);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const imgData = ctx.createImageData(size, size);
    for (let i = 0; i < imgData.data.length; i += 4) {
        const value = Math.random() * 255;
        imgData.data[i] = value;
        imgData.data[i + 1] = value;
        imgData.data[i + 2] = value;
        imgData.data[i + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);

    ctx.globalAlpha = 0.35;
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, size, size);
    ctx.globalAlpha = 1;

    return canvas;
}

const BASE_PANEL_MAP_KEY = 'sciFi/panel/map';
const BASE_PANEL_EMISSIVE_KEY = 'sciFi/panel/emissive';
const BASE_PANEL_ROUGHNESS_KEY = 'sciFi/panel/roughness';

function getPanelMapTexture() {
    return registerTexture(BASE_PANEL_MAP_KEY, () => drawNeonGridBackground());
}

function getPanelEmissiveTexture() {
    const tex = registerTexture(BASE_PANEL_EMISSIVE_KEY, () => drawEmissiveScanTexture());
    if (tex) {
        tex.wrapS = THREE.RepeatWrapping;
        tex.wrapT = THREE.RepeatWrapping;
    }
    return tex;
}

function getPanelRoughnessTexture() {
    const tex = registerTexture(BASE_PANEL_ROUGHNESS_KEY, () => drawNoiseTexture());
    if (tex) {
        tex.wrapS = THREE.RepeatWrapping;
        tex.wrapT = THREE.RepeatWrapping;
    }
    return tex;
}

function applyShaderEnhancements(material, { emissiveColor, scanSpeed = 2.5, scanStrength = 0.35, fresnelPower = 2.5, fresnelIntensity = 0.6 } = {}) {
    const glowColor = new THREE.Color(emissiveColor ?? 0x00ffff);
    const startTime = (typeof performance !== 'undefined') ? performance.now() : Date.now();

    material.onBeforeCompile = (shader) => {
        shader.uniforms.uTime = { value: 0 };
        shader.uniforms.uGlowColor = { value: glowColor.clone() };
        shader.uniforms.uScanSpeed = { value: scanSpeed };
        shader.uniforms.uScanStrength = { value: scanStrength };
        shader.uniforms.uFresnelPower = { value: fresnelPower };
        shader.uniforms.uFresnelIntensity = { value: fresnelIntensity };

        material.userData.shaderUniforms = shader.uniforms;

        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <lights_fragment_begin>',
            `#include <lights_fragment_begin>\n\n` +
            'float viewDot = abs(dot(normalize(vNormal), normalize(-vViewPosition)));\n' +
            'float fresnelTerm = pow(1.0 - saturate(viewDot), uFresnelPower);\n' +
            'totalEmissiveRadiance += uGlowColor * fresnelTerm * uFresnelIntensity;\n' +
            'float scan = 0.5 + 0.5 * sin(vUv.y * 40.0 + uTime * uScanSpeed);\n' +
            'totalEmissiveRadiance += uGlowColor * scan * uScanStrength;\n'
        );
    };

    material.onBeforeRender = function () {
        const uniforms = material.userData.shaderUniforms;
        if (!uniforms) return;
        const now = (typeof performance !== 'undefined') ? performance.now() : Date.now();
        uniforms.uTime.value = (now - startTime) / 1000;
    };

    material.customProgramCacheKey = () => `sciFiPanel-${scanSpeed}-${scanStrength}-${fresnelPower}-${fresnelIntensity}-${glowColor.getHexString()}`;
}

export function createSciFiPanelMaterial({
    baseColor = 0xffffff,
    emissiveColor = 0x00d2ff,
    opacity = 0.92,
    doubleSided = false,
    mapRepeat = new THREE.Vector2(3, 2),
    scanStrength = 0.35,
    scanSpeed = 2.5,
    fresnelPower = 2.5,
    fresnelIntensity = 0.6
} = {}) {
    const panelMap = getPanelMapTexture();
    const panelEmissive = getPanelEmissiveTexture();
    const panelRoughness = getPanelRoughnessTexture();

    if (panelMap) {
        panelMap.repeat.copy(mapRepeat);
    }
    if (panelEmissive) {
        panelEmissive.repeat.copy(new THREE.Vector2(mapRepeat.x * 1.5, mapRepeat.y * 3));
    }
    if (panelRoughness) {
        panelRoughness.repeat.copy(new THREE.Vector2(mapRepeat.x * 2, mapRepeat.y * 2));
    }

    const material = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(baseColor),
        map: panelMap || null,
        emissive: new THREE.Color(emissiveColor).multiplyScalar(0.25),
        emissiveMap: panelEmissive || null,
        emissiveIntensity: 0.65,
        metalness: 0.65,
        roughness: 0.2,
        roughnessMap: panelRoughness || null,
        clearcoat: 0.75,
        clearcoatRoughness: 0.1,
        sheen: 0.35,
        sheenColor: new THREE.Color(emissiveColor).lerp(new THREE.Color(baseColor), 0.4),
        transmission: 0.05,
        thickness: 1.25,
        envMapIntensity: 1.2,
        opacity,
        transparent: true,
        side: doubleSided ? THREE.DoubleSide : THREE.FrontSide
    });

    applyShaderEnhancements(material, { emissiveColor, scanSpeed, scanStrength, fresnelPower, fresnelIntensity });

    return material;
}

export function createSciFiOrbMaterial(color) {
    const baseColor = new THREE.Color(color || 0x4fd1ff);
    const emissiveColor = baseColor.clone().lerp(new THREE.Color(0xffffff), 0.25);

    const material = new THREE.MeshPhysicalMaterial({
        color: baseColor,
        emissive: emissiveColor,
        emissiveIntensity: 0.85,
        metalness: 0.3,
        roughness: 0.1,
        clearcoat: 0.5,
        clearcoatRoughness: 0.1,
        transmission: 0.4,
        thickness: 0.9,
        envMapIntensity: 1.1,
        opacity: 0.9,
        transparent: true,
        side: THREE.FrontSide
    });

    const glowColor = emissiveColor.clone();
    const startTime = (typeof performance !== 'undefined') ? performance.now() : Date.now();

    material.onBeforeCompile = (shader) => {
        shader.uniforms.uTime = { value: 0 };
        shader.uniforms.uGlowColor = { value: glowColor };
        material.userData.shaderUniforms = shader.uniforms;
        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <lights_fragment_begin>',
            `#include <lights_fragment_begin>\n` +
            'float pulse = 0.5 + 0.5 * sin(uTime * 3.0);\n' +
            'totalEmissiveRadiance += uGlowColor * pulse * 0.35;\n'
        );
    };

    material.onBeforeRender = function () {
        const uniforms = material.userData.shaderUniforms;
        if (!uniforms) return;
        const now = (typeof performance !== 'undefined') ? performance.now() : Date.now();
        uniforms.uTime.value = (now - startTime) / 1000;
    };

    material.customProgramCacheKey = () => `sciFiOrb-${glowColor.getHexString()}`;

    return material;
}

export function updateMaterialUniformSpeed(material, speed = 1) {
    if (!material || !material.userData || !material.userData.shaderUniforms) return;
    if (material.userData.shaderUniforms.uScanSpeed) {
        material.userData.shaderUniforms.uScanSpeed.value = speed;
    }
}
