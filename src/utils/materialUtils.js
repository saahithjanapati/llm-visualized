import * as THREE from 'three';

export function applyPhysicalMaterialsToScene(scene, enabled = true) {
    if (!scene || typeof scene.traverse !== 'function') return;

    const usePhysical = !!enabled;
    const fromCtor = usePhysical ? THREE.MeshStandardMaterial : THREE.MeshPhysicalMaterial;
    const toCtor = usePhysical ? THREE.MeshPhysicalMaterial : THREE.MeshStandardMaterial;

    scene.traverse((obj) => {
        if (!obj || !obj.isMesh) return;
        const swapMat = (mat) => {
            if (!(mat instanceof fromCtor)) return mat;
            const params = {
                color: mat.color?.clone?.(),
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

export default applyPhysicalMaterialsToScene;
