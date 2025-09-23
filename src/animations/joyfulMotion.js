import * as THREE from 'three';

const clamp01 = (v) => THREE.MathUtils.clamp(v, 0, 1);

export const JoyEasing = {
    easeInOutSine: (k) => 0.5 * (1 - Math.cos(Math.PI * clamp01(k))),
    easeOutBack: (k) => {
        const t = clamp01(k) - 1;
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return 1 + c3 * t * t * t + c1 * t * t;
    },
    easeOutElastic: (k) => {
        const t = clamp01(k);
        if (t === 0 || t === 1) return t;
        const c4 = (2 * Math.PI) / 3;
        return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
    },
};

export function computeArcOffset(progress, amplitude = 10) {
    return Math.sin(Math.PI * clamp01(progress)) * amplitude;
}

export function applySquashStretchFromProgress(object3D, progress, options = {}) {
    if (!object3D || !object3D.scale) return;
    const p = clamp01(progress);
    const intensity = options.intensity ?? 0.2;
    const rotationAmplitude = options.rotationAmplitude ?? 0.15;

    if (!object3D.userData) object3D.userData = {};
    let baseScale = options.baseScale;
    if (!baseScale) {
        if (object3D.userData.__joyfulBaseScale) {
            baseScale = object3D.userData.__joyfulBaseScale;
        } else {
            baseScale = object3D.scale.clone();
            object3D.userData.__joyfulBaseScale = baseScale.clone();
        }
    } else if (!object3D.userData.__joyfulBaseScale) {
        object3D.userData.__joyfulBaseScale = baseScale.clone();
    }

    const stretch = 1 + Math.sin(p * Math.PI) * intensity;
    const squash = 1 / Math.max(0.0001, Math.sqrt(stretch));
    object3D.scale.set(baseScale.x * squash, baseScale.y * stretch, baseScale.z * squash);

    if (object3D.rotation) {
        object3D.rotation.z = rotationAmplitude * Math.sin(p * Math.PI);
    }
}

export function resetJoyfulTransform(object3D, baseScale = null) {
    if (!object3D || !object3D.scale) return;
    if (!object3D.userData) object3D.userData = {};
    let targetScale = baseScale;
    if (!targetScale) {
        if (object3D.userData.__joyfulBaseScale) {
            targetScale = object3D.userData.__joyfulBaseScale;
        }
    }
    if (targetScale) {
        object3D.scale.copy(targetScale);
    }
    if (object3D.rotation) {
        object3D.rotation.z = 0;
    }
}

export function createJoyfulRiseTween(object3D, targetY, options = {}) {
    if (typeof TWEEN === 'undefined') return null;
    if (!object3D || !object3D.position) return null;

    const startY = object3D.position.y;
    const distance = Math.abs(targetY - startY);
    const baseDuration = options.duration ?? Math.max(450, distance * 18);
    const anticipation = options.anticipation ?? Math.min(24 + distance * 0.12, Math.max(12, distance * 0.25));
    const overshoot = options.overshoot ?? Math.min(36 + distance * 0.18, Math.max(18, distance * 0.45));

    const anticipationY = startY - anticipation;
    const overshootY = targetY + overshoot;
    const intensity = options.intensity ?? 0.24;
    const rotationAmplitude = options.rotationAmplitude ?? 0.18;

    const baseScale = object3D.scale ? object3D.scale.clone() : new THREE.Vector3(1, 1, 1);

    const state = { progress: 0 };
    const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
    const onComplete = typeof options.onComplete === 'function' ? options.onComplete : null;

    const applyState = (progress) => {
        const p = clamp01(progress);
        let y;
        if (p < 0.2) {
            const local = p / 0.2;
            y = THREE.MathUtils.lerp(startY, anticipationY, JoyEasing.easeInOutSine(local));
        } else if (p < 0.7) {
            const local = (p - 0.2) / 0.5;
            y = THREE.MathUtils.lerp(anticipationY, overshootY, JoyEasing.easeOutBack(local));
        } else {
            const local = (p - 0.7) / 0.3;
            y = THREE.MathUtils.lerp(overshootY, targetY, JoyEasing.easeOutElastic(local));
        }
        object3D.position.y = y;
        applySquashStretchFromProgress(object3D, p, { intensity, rotationAmplitude, baseScale });
        if (onProgress) onProgress(p);
    };

    const tween = new TWEEN.Tween(state)
        .to({ progress: 1 }, baseDuration)
        .easing(JoyEasing.easeInOutSine)
        .onUpdate(() => applyState(state.progress))
        .onComplete(() => {
            object3D.position.y = targetY;
            resetJoyfulTransform(object3D, baseScale);
            if (onProgress) onProgress(1);
            if (onComplete) onComplete();
        })
        .start();

    return tween;
}
