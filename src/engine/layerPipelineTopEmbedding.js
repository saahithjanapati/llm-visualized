import * as THREE from 'three';

export function calculateTopEmbeddingTargets({
    engineScene = null,
    lastLayer = null,
    mlpDownHeight = 0,
    embedHeight = 0,
    embedInset = 5,
    topEmbedGap = 0,
    topEmbedAdjust = 0,
    maxRiseFraction = 1
} = {}) {
    if (!lastLayer || !lastLayer.root) return { targetYLocal: null, exitYLocal: null };

    let targetYLocal = null;
    let exitYLocal = null;
    try {
        let topEmbedObj = null;
        if (engineScene && typeof engineScene.traverse === 'function') {
            engineScene.traverse((obj) => {
                if (topEmbedObj) return;
                const label = obj && obj.userData ? obj.userData.label : '';
                if (
                    label === 'Vocab Embedding (Top)'
                    || label === 'Vocabulary Embedding (Top)'
                    || label === 'Vocab Unembedding'
                    || label === 'Vocabulary Unembedding'
                ) {
                    topEmbedObj = obj;
                }
            });
        }
        if (topEmbedObj) {
            const centerWorld = new THREE.Vector3();
            topEmbedObj.getWorldPosition(centerWorld);
            const entryWorldY = centerWorld.y - embedHeight / 2 + embedInset;
            const exitWorldY = centerWorld.y + embedHeight / 2 - embedInset;
            const entryLocalVec = new THREE.Vector3(0, entryWorldY, 0);
            lastLayer.root.worldToLocal(entryLocalVec);
            targetYLocal = entryLocalVec.y;
            const exitLocalVec = new THREE.Vector3(0, exitWorldY, 0);
            lastLayer.root.worldToLocal(exitLocalVec);
            exitYLocal = exitLocalVec.y;
        }
    } catch (_) {
        // Fallback to formula below.
    }

    if (targetYLocal == null) {
        const towerTopYLocal = lastLayer.mlpDown.group.position.y + mlpDownHeight / 2;
        const topVocabCenterYLocal = towerTopYLocal + topEmbedGap + embedHeight / 2 + topEmbedAdjust;
        targetYLocal = topVocabCenterYLocal - embedHeight / 2 + embedInset;
        exitYLocal = topVocabCenterYLocal + embedHeight / 2 - embedInset;
    }

    const riseFracRaw = Number.isFinite(maxRiseFraction) ? maxRiseFraction : 1;
    const riseFrac = Math.max(0, Math.min(1, riseFracRaw));
    if (Number.isFinite(targetYLocal) && Number.isFinite(exitYLocal) && riseFrac < 1) {
        const maxRise = embedHeight * riseFrac;
        const cappedExit = targetYLocal + maxRise;
        if (exitYLocal > cappedExit) exitYLocal = cappedExit;
    }

    if (Number.isFinite(exitYLocal) && Number.isFinite(targetYLocal) && exitYLocal < targetYLocal) {
        exitYLocal = targetYLocal;
    }

    return { targetYLocal, exitYLocal };
}

export function findTopLayerNormInfo({ engineScene = null, lastLayer = null, lnHeight = 0 } = {}) {
    if (!lastLayer || !lastLayer.root || !engineScene || typeof engineScene.traverse !== 'function') {
        return null;
    }
    let lnTopGroup = null;
    try {
        engineScene.traverse(obj => {
            if (!lnTopGroup && obj && obj.userData && obj.userData.label === 'LayerNorm (Top)') {
                lnTopGroup = obj;
            }
        });
    } catch (_) {
        return null;
    }

    if (!lnTopGroup) return null;
    const lnCenterWorld = new THREE.Vector3();
    lnTopGroup.getWorldPosition(lnCenterWorld);
    const lnCenterLocal = lnCenterWorld.clone();
    lastLayer.root.worldToLocal(lnCenterLocal);
    const lnCenterY = lnCenterLocal.y;
    const lnBottomY = lnCenterY - lnHeight / 2;
    return { lnTopGroup, lnCenterY, lnBottomY };
}

export function activateLayerNormColor(lnTopGroup, {
    emissiveIntensity = 0.5,
    scaleEmissiveIntensity = (value) => value
} = {}) {
    if (!lnTopGroup || typeof lnTopGroup.traverse !== 'function') return;
    const white = new THREE.Color(0xffffff);
    const resolvedEmissive = scaleEmissiveIntensity(emissiveIntensity);
    lnTopGroup.traverse(obj => {
        if (obj.isMesh && obj.material) {
            const apply = (mat) => {
                mat.color.copy(white);
                mat.emissive.copy(white);
                mat.emissiveIntensity = resolvedEmissive;
                mat.transparent = false;
                mat.opacity = 1.0;
            };
            if (Array.isArray(obj.material)) obj.material.forEach(apply); else apply(obj.material);
        }
    });
}
