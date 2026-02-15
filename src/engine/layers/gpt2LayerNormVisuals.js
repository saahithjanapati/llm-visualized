import * as THREE from 'three';
import { applyLayerNormMaterial } from './gpt2LayerUtils.js';

function clamp01(value) {
    return Math.min(1, Math.max(0, value));
}

export function updateLayerNormVisualState({
    layerNorm,
    targetColor,
    lockedColor,
    isColorLocked,
    materialState,
    highestVecY,
    anyVectorInNorm,
    bottomY,
    midY,
    topY,
    exitTransitionRange,
    inactiveColor,
    activeColor,
    finalColor,
    opaqueOpacity,
    activeOpacity,
    skipActive,
    skipColorLerpAlpha,
    applyWhenInactive = false,
}) {
    if (!targetColor || !lockedColor || !materialState) {
        return { colorLocked: !!isColorLocked };
    }

    targetColor.copy(inactiveColor);
    let targetOpacity = opaqueOpacity;
    const hasActiveVector = anyVectorInNorm && highestVecY > -Infinity;

    if (hasActiveVector) {
        if (highestVecY >= bottomY && highestVecY < midY) {
            const t = (highestVecY - bottomY) / (midY - bottomY);
            targetColor.lerpColors(inactiveColor, activeColor, t);
            targetOpacity = THREE.MathUtils.lerp(opaqueOpacity, activeOpacity, t);
        } else if (highestVecY >= midY && highestVecY < topY) {
            targetColor.copy(activeColor);
            targetOpacity = activeOpacity;
        } else if (highestVecY >= topY) {
            const tRaw = (highestVecY - topY) / exitTransitionRange;
            const t = clamp01(tRaw);
            targetColor.lerpColors(activeColor, finalColor, t);
            targetOpacity = THREE.MathUtils.lerp(activeOpacity, opaqueOpacity, t);
        }
    }

    let colorLocked = !!isColorLocked;
    if (!colorLocked && highestVecY >= topY + exitTransitionRange) {
        colorLocked = true;
        lockedColor.copy(finalColor);
    }

    if (colorLocked) {
        targetColor.copy(lockedColor);
        targetOpacity = opaqueOpacity;
    }

    const shouldApply = applyWhenInactive || hasActiveVector || colorLocked;
    if (!shouldApply) {
        return { colorLocked };
    }

    if (skipActive && materialState.initialized) {
        const smoothAlpha = skipColorLerpAlpha;
        if (smoothAlpha > 0 && smoothAlpha < 1) {
            targetColor.lerpColors(materialState.color, targetColor, smoothAlpha);
            targetOpacity = THREE.MathUtils.lerp(materialState.opacity, targetOpacity, smoothAlpha);
        }
    }

    applyLayerNormMaterial(layerNorm && layerNorm.group, targetColor, targetOpacity, materialState);
    return { colorLocked };
}

