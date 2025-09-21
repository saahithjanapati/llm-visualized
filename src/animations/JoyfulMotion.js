import * as THREE from 'three';

function gatherMaterials(object3D, list = []) {
    if (!object3D) return list;
    const visit = (obj) => {
        if (!obj) return;
        if (obj.isMesh && obj.material) {
            if (Array.isArray(obj.material)) {
                obj.material.forEach(mat => mat && list.push(mat));
            } else {
                list.push(obj.material);
            }
        }
        if (obj.children && obj.children.length) {
            obj.children.forEach(child => visit(child));
        }
    };
    visit(object3D);
    return list;
}

function easeOutBack(t, amount = 1.70158) {
    const c3 = amount + 1;
    const tt = t - 1;
    return 1 + c3 * tt * tt * tt + amount * tt * tt;
}

export function pulseMaterialEmissive(target, {
    color = new THREE.Color(0xffffff),
    intensity = 0.75,
    duration = 600,
    delay = 0,
    yoyo = true,
    repeat = 1
} = {}) {
    if (!target || typeof TWEEN === 'undefined') return null;

    const mats = gatherMaterials(target, []);
    if (!mats.length) return null;

    const baseState = mats.map(mat => ({
        emissive: mat.emissive ? mat.emissive.clone() : new THREE.Color(0x000000),
        intensity: typeof mat.emissiveIntensity === 'number' ? mat.emissiveIntensity : 0
    }));

    const tweenState = { factor: 0 };
    const totalRepeat = Math.max(0, repeat);

    const tween = new TWEEN.Tween(tweenState)
        .to({ factor: 1 }, duration)
        .delay(delay)
        .easing(TWEEN.Easing.Sine.InOut)
        .yoyo(yoyo)
        .repeat(totalRepeat)
        .onUpdate(() => {
            mats.forEach((mat, idx) => {
                if (!mat) return;
                if (!mat.emissive) {
                    mat.emissive = new THREE.Color(0x000000);
                }
                const base = baseState[idx];
                mat.emissive.copy(base.emissive).lerp(color, tweenState.factor);
                mat.emissiveIntensity = THREE.MathUtils.lerp(base.intensity, intensity, tweenState.factor);
            });
        })
        .onComplete(() => {
            mats.forEach((mat, idx) => {
                const base = baseState[idx];
                if (mat && base) {
                    mat.emissive.copy(base.emissive);
                    mat.emissiveIntensity = base.intensity;
                }
            });
        })
        .start();

    return tween;
}

export function animateJoyfulRise(object3D, targetY, {
    speed = 100,
    duration = null,
    minDuration = 320,
    anticipationOffset,
    overshoot,
    delay = 0,
    enableSquashStretch = true,
    squashAmount = 0.22,
    stretchAmount = 0.35,
    enableSwing = true,
    swingAmount = 0.18,
    swingAxis = 'z',
    emissivePulse,
    easeUp,
    easeSettle,
    anticipationDurationRatio = 0.25,
    settleDurationRatio = 0.2,
    onUpdate,
    onComplete
} = {}) {
    if (!object3D) return null;
    const currentY = object3D.position.y || 0;
    if (targetY == null || currentY >= targetY - 0.0001) {
        if (typeof targetY === 'number') {
            object3D.position.y = targetY;
        }
        if (typeof onUpdate === 'function') onUpdate({ y: object3D.position.y });
        if (typeof onComplete === 'function') onComplete();
        return null;
    }

    const distance = Math.max(0, targetY - currentY);
    const computedDuration = duration != null
        ? duration
        : Math.max(minDuration, (distance / Math.max(speed, 0.0001)) * 1000);

    const anticipationDefault = -Math.min(60, Math.max(12, distance * 0.08));
    const overshootDefault = Math.min(120, Math.max(16, distance * 0.12));

    const anticipation = anticipationOffset != null ? anticipationOffset : anticipationDefault;
    const overshootAmount = overshoot != null ? overshoot : overshootDefault;

    const anticipationDuration = Math.max(0, computedDuration * anticipationDurationRatio);
    const settleDuration = Math.max(0, computedDuration * settleDurationRatio);
    const riseDuration = Math.max(1, computedDuration - anticipationDuration - settleDuration);

    const totalTimeline = anticipationDuration + riseDuration + settleDuration;
    const actualDelay = Math.max(0, delay);

    if (typeof TWEEN === 'undefined') {
        object3D.position.y = targetY;
        if (typeof onUpdate === 'function') onUpdate({ y: object3D.position.y });
        if (typeof onComplete === 'function') onComplete();
        return null;
    }

    const state = { y: currentY };
    const applyState = () => {
        object3D.position.y = state.y;
        if (typeof onUpdate === 'function') onUpdate({ y: state.y });
    };

    const tweens = [];
    const overshootTarget = targetY + overshootAmount;

    const createTween = (targetValue, dur, easing) => new TWEEN.Tween(state)
        .to({ y: targetValue }, Math.max(1, dur))
        .easing(easing)
        .onUpdate(applyState);

    let firstTween = null;
    let previousTween = null;

    if (anticipationDuration > 1 && Math.abs(anticipation) > 0.5) {
        const anticipationTween = createTween(currentY + anticipation, anticipationDuration, TWEEN.Easing.Quadratic.InOut)
            .delay(actualDelay);
        firstTween = anticipationTween;
        previousTween = anticipationTween;
        tweens.push(anticipationTween);
    }

    const riseTween = createTween(overshootTarget, riseDuration, easeUp || TWEEN.Easing.Cubic.Out);
    if (!firstTween) {
        riseTween.delay(actualDelay);
        firstTween = riseTween;
    } else if (previousTween) {
        previousTween.chain(riseTween);
    }
    previousTween = riseTween;
    tweens.push(riseTween);

    const settleTween = createTween(targetY, settleDuration, easeSettle || ((typeof TWEEN !== 'undefined' && TWEEN.Easing.Back)
        ? TWEEN.Easing.Back.Out
        : TWEEN.Easing.Cubic.InOut));
    settleTween.onComplete(() => {
        object3D.position.y = targetY;
        if (typeof onComplete === 'function') onComplete();
    });
    previousTween.chain(settleTween);
    tweens.push(settleTween);

    if (firstTween) {
        firstTween.start();
    }

    if (enableSquashStretch && object3D.scale) {
        const baseScale = object3D.scale.clone();
        const scaleState = { t: 0 };
        new TWEEN.Tween(scaleState)
            .to({ t: 1 }, totalTimeline)
            .delay(actualDelay)
            .easing(TWEEN.Easing.Sine.InOut)
            .onUpdate(() => {
                const pulse = Math.sin(scaleState.t * Math.PI);
                const squash = 1 - pulse * squashAmount;
                const stretch = 1 + pulse * stretchAmount;
                object3D.scale.set(
                    baseScale.x * squash,
                    baseScale.y * stretch,
                    baseScale.z * squash
                );
            })
            .onComplete(() => {
                object3D.scale.copy(baseScale);
            })
            .start();
    }

    if (enableSwing && object3D.rotation) {
        const baseRot = object3D.rotation.clone();
        const swingState = { t: 0 };
        new TWEEN.Tween(swingState)
            .to({ t: 1 }, totalTimeline + actualDelay)
            .delay(actualDelay)
            .easing(TWEEN.Easing.Sine.InOut)
            .onUpdate(() => {
                const sway = Math.sin(swingState.t * Math.PI) * swingAmount;
                object3D.rotation.copy(baseRot);
                switch (swingAxis) {
                    case 'x':
                        object3D.rotation.x = baseRot.x + sway;
                        break;
                    case 'y':
                        object3D.rotation.y = baseRot.y + sway;
                        break;
                    default:
                        object3D.rotation.z = baseRot.z + sway;
                }
            })
            .onComplete(() => {
                object3D.rotation.copy(baseRot);
            })
            .start();
    }

    if (emissivePulse) {
        const { color, intensity, pulseDuration, pulseDelay, repeat } = emissivePulse;
        pulseMaterialEmissive(object3D, {
            color: color instanceof THREE.Color ? color : (color ? new THREE.Color(color) : new THREE.Color(0xffffff)),
            intensity: typeof intensity === 'number' ? intensity : 0.75,
            duration: typeof pulseDuration === 'number' ? pulseDuration : 500,
            delay: typeof pulseDelay === 'number' ? pulseDelay + actualDelay : actualDelay,
            repeat: typeof repeat === 'number' ? repeat : 1
        });
    }

    return tweens[tweens.length - 1];
}

export function addJoyfulArc(progress, amplitude = 20, phaseOffset = 0) {
    const clamped = THREE.MathUtils.clamp(progress, 0, 1);
    const eased = easeOutBack(clamped, 1.2);
    const arc = Math.sin((eased + phaseOffset) * Math.PI);
    return arc * amplitude;
}
