import * as THREE from 'three';
import { MHSAAnimation } from '../src/animations/MHSAAnimation.js';
import { loadPrecomputedGeometries } from '../src/utils/precomputedGeometryLoader.js';
import { LayerPipeline } from '../src/engine/LayerPipeline.js';
import { setPlaybackSpeed, USE_PHYSICAL_MATERIALS } from '../src/utils/constants.js';

export async function initPipeline() {
    await loadPrecomputedGeometries('../precomputed_components.glb');
    try { setPlaybackSpeed('fast'); } catch (_) { /* no-op */ }

    MHSAAnimation.ENABLE_SELF_ATTENTION = true;

    const gptCanvas = document.getElementById('gptCanvas');
    const NUM_LAYERS = 12;
    const camPos = new THREE.Vector3(0, 11000, 16000);
    const camTarget = new THREE.Vector3(0, 9000, 0);

    const pipeline = new LayerPipeline(gptCanvas, NUM_LAYERS, {
        cameraPosition: camPos,
        cameraTarget: camTarget
    });

    gptCanvas.style.display = 'block';
    try {
        const eng = pipeline.engine;
        eng.renderer.shadowMap.enabled = false;
        eng.scene.traverse((obj) => {
            if (obj.isMesh) {
                obj.castShadow = false;
                obj.receiveShadow = false;
            }
            if (obj.isLight) {
                obj.castShadow = false;
            }
        });
    } catch (_) { /* no-op */ }

    function applyPhysicalMaterial(enabled) {
        if (!pipeline?.engine?.scene) return;
        const fromCtor = enabled ? THREE.MeshStandardMaterial : THREE.MeshPhysicalMaterial;
        const toCtor = enabled ? THREE.MeshPhysicalMaterial : THREE.MeshStandardMaterial;
        pipeline.engine.scene.traverse((obj) => {
            if (!obj.isMesh) return;
            const swapMat = (mat) => {
                if (!(mat instanceof fromCtor)) return mat;
                const params = {
                    color: mat.color?.clone(),
                    metalness: mat.metalness ?? 0,
                    roughness: mat.roughness ?? 1,
                    transparent: mat.transparent,
                    opacity: mat.opacity,
                    emissive: mat.emissive?.clone?.(),
                    emissiveIntensity: mat.emissiveIntensity ?? 0,
                    flatShading: mat.flatShading,
                    side: mat.side,
                };
                const newMat = new toCtor(params);
                mat.dispose();
                return newMat;
            };
            if (Array.isArray(obj.material)) {
                obj.material = obj.material.map(swapMat);
            } else if (obj.material) {
                obj.material = swapMat(obj.material);
            }
        });
    }
    try { applyPhysicalMaterial(USE_PHYSICAL_MATERIALS); } catch (_) {}

    return { pipeline, NUM_LAYERS, gptCanvas };
}
