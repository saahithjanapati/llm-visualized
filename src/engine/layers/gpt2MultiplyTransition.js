import * as THREE from 'three';

const MULTIPLY_TRANSITION_DURATION_MS = 160;
const MULTIPLY_SOURCE_SHRINK = 0.94;
const MULTIPLY_RESULT_POP_SCALE = 1.16;
const MULTIPLY_RESULT_POP_EXPAND_MS = 95;
const MULTIPLY_RESULT_POP_SETTLE_MS = 115;

function defaultSetVectorOpacity(vec, opacity) {
    if (!vec || !vec.mesh || !vec.mesh.material) return;
    const clampedOpacity = THREE.MathUtils.clamp(opacity, 0, 1);
    const materials = Array.isArray(vec.mesh.material) ? vec.mesh.material : [vec.mesh.material];
    materials.forEach(mat => {
        if (!mat) return;
        const shouldBeTransparent = clampedOpacity < 0.999;
        if (mat.transparent !== shouldBeTransparent) {
            mat.transparent = shouldBeTransparent;
            mat.needsUpdate = true;
        }
        if (mat.opacity !== clampedOpacity) {
            mat.opacity = clampedOpacity;
        }
        if (mat.depthWrite === shouldBeTransparent) {
            mat.depthWrite = !shouldBeTransparent;
            mat.needsUpdate = true;
        }
        if (!shouldBeTransparent && mat.depthWrite !== true) {
            mat.depthWrite = true;
            mat.needsUpdate = true;
        }
    });
}

function defaultSetScaleParamVisible(scaleParam, visible) {
    if (scaleParam && scaleParam.group) {
        scaleParam.group.visible = visible;
    }
}

export function animatePrismMultiplyTransition({
    sourceVec,
    multResult,
    scaleParam = null,
    setScaleParamVisible = null,
    setVectorOpacity = null,
    emitProgress = null,
    skipToEndActive = false,
    instant = false,
    onComplete = null
} = {}) {
    const applyScaleParamVisibility = typeof setScaleParamVisible === 'function'
        ? (visible) => setScaleParamVisible(scaleParam, visible)
        : (visible) => defaultSetScaleParamVisible(scaleParam, visible);
    const applyVectorOpacity = typeof setVectorOpacity === 'function'
        ? setVectorOpacity
        : defaultSetVectorOpacity;
    const progress = typeof emitProgress === 'function' ? emitProgress : () => {};

    const finalizeVisibility = () => {
        applyScaleParamVisibility(false);
        if (sourceVec && sourceVec.group) {
            sourceVec.group.visible = false;
            if (sourceVec.group.parent) {
                sourceVec.group.parent.remove(sourceVec.group);
            }
        }
        if (multResult && multResult.group) {
            multResult.group.visible = true;
            multResult.group.scale.set(1, 1, 1);
            applyVectorOpacity(multResult, 1);
        }
    };

    const finish = () => {
        if (typeof onComplete === 'function') onComplete();
    };

    const pulseMultiplyResult = () => {
        if (!multResult || !multResult.group || skipToEndActive || typeof TWEEN === 'undefined') {
            finish();
            return;
        }

        const baseScale = multResult.group.scale.clone();
        const pulseState = { s: 1 };
        const applyPulseScale = () => {
            multResult.group.scale.set(
                baseScale.x * pulseState.s,
                baseScale.y * pulseState.s,
                baseScale.z * pulseState.s
            );
            progress();
        };

        new TWEEN.Tween(pulseState)
            .to({ s: MULTIPLY_RESULT_POP_SCALE }, MULTIPLY_RESULT_POP_EXPAND_MS)
            .easing(TWEEN.Easing.Back.Out)
            .onUpdate(applyPulseScale)
            .onComplete(() => {
                new TWEEN.Tween(pulseState)
                    .to({ s: 1 }, MULTIPLY_RESULT_POP_SETTLE_MS)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(applyPulseScale)
                    .onComplete(() => {
                        multResult.group.scale.copy(baseScale);
                        progress();
                        finish();
                    })
                    .start();
            })
            .start();
    };

    if (instant) {
        finalizeVisibility();
        pulseMultiplyResult();
        return;
    }

    if (!multResult || !multResult.group) {
        finalizeVisibility();
        finish();
        return;
    }

    if (!sourceVec || !sourceVec.group) {
        finalizeVisibility();
        pulseMultiplyResult();
        return;
    }

    if (skipToEndActive || typeof TWEEN === 'undefined') {
        finalizeVisibility();
        finish();
        return;
    }

    const sourceStartScale = sourceVec.group.scale.clone();
    const sourceEndScale = sourceStartScale.clone().multiplyScalar(MULTIPLY_SOURCE_SHRINK);

    multResult.group.visible = false;
    multResult.group.scale.set(1, 1, 1);
    applyVectorOpacity(sourceVec, 1);
    applyVectorOpacity(multResult, 1);
    applyScaleParamVisibility(false);

    const tweenState = { t: 0 };
    new TWEEN.Tween(tweenState)
        .to({ t: 1 }, MULTIPLY_TRANSITION_DURATION_MS)
        .easing(TWEEN.Easing.Quadratic.Out)
        .onUpdate(() => {
            const t = THREE.MathUtils.clamp(tweenState.t, 0, 1);
            sourceVec.group.scale.lerpVectors(sourceStartScale, sourceEndScale, t);
            progress();
        })
        .onComplete(() => {
            finalizeVisibility();
            pulseMultiplyResult();
        })
        .start();
}
