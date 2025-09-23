import * as THREE from 'three';

function getTween() {
    // Helper to safely access the global TWEEN namespace when available.
    return typeof TWEEN !== 'undefined' ? TWEEN : null;
}

function clamp01(value) {
    return Math.min(1, Math.max(0, value));
}

/**
 * Creates a quick anticipation tween that pulls the object slightly backwards
 * before a larger movement. Applies a touch of squash and stretch for extra
 * charm.
 * @param {THREE.Object3D} object3d
 * @param {'x'|'y'|'z'} axis
 * @param {number} offset
 * @param {number} duration
 * @param {object} [opts]
 * @returns {TWEEN.Tween|null}
 */
export function createAnticipationTween(object3d, axis, offset, duration, opts = {}) {
    const tweenNs = getTween();
    if (!tweenNs || !object3d || !object3d.position || typeof object3d.position[axis] !== 'number') {
        return null;
    }

    const targetOffset = Math.abs(offset);
    if (targetOffset === 0) return null;

    const baseScale = object3d.scale ? object3d.scale.clone() : null;
    const startValue = object3d.position[axis];
    const endValue = startValue - Math.sign(offset) * targetOffset;
    const squashAmount = typeof opts.squashAmount === 'number' ? opts.squashAmount : 0.1;

    const state = { value: startValue };
    const tween = new tweenNs.Tween(state)
        .to({ value: endValue }, Math.max(50, duration || 200))
        .easing(opts.easing || tweenNs.Easing.Cubic.Out)
        .onUpdate(() => {
            object3d.position[axis] = state.value;
            if (baseScale && squashAmount) {
                const travelled = startValue - state.value;
                const progress = clamp01(travelled / targetOffset);
                const squash = 1 + squashAmount * progress;
                const stretch = 1 - squashAmount * 0.7 * progress;
                object3d.scale.set(
                    baseScale.x * stretch,
                    baseScale.y * squash,
                    baseScale.z * stretch
                );
            }
            if (typeof opts.onProgress === 'function') {
                const progress = clamp01((startValue - state.value) / targetOffset);
                opts.onProgress(progress);
            }
        })
        .onComplete(() => {
            if (baseScale) {
                object3d.scale.copy(baseScale);
            }
            if (typeof opts.onComplete === 'function') {
                opts.onComplete();
            }
        });

    if (typeof opts.delay === 'number') {
        tween.delay(Math.max(0, opts.delay));
    }

    return tween;
}

/**
 * Animates an object along a soft bezier arc between start and end positions.
 * Applies secondary squash/stretch and a gentle rotation for appeal.
 * @param {THREE.Object3D} object3d
 * @param {object} options
 * @returns {TWEEN.Tween|null}
 */
export function createArcTween(object3d, options = {}) {
    const tweenNs = getTween();
    if (!tweenNs || !object3d || !object3d.position) return null;

    const startVec = (options.start instanceof THREE.Vector3)
        ? options.start.clone()
        : object3d.position.clone();
    const endVec = (options.end instanceof THREE.Vector3)
        ? options.end.clone()
        : object3d.position.clone();

    const control = new THREE.Vector3(
        (startVec.x + endVec.x) / 2,
        (startVec.y + endVec.y) / 2,
        (startVec.z + endVec.z) / 2
    );
    control.y += options.lift != null ? options.lift : 0;
    if (options.liftZ) control.z += options.liftZ;

    const duration = Math.max(50, options.duration || 1000);
    const baseScale = object3d.scale ? object3d.scale.clone() : null;
    const baseRotation = object3d.rotation ? object3d.rotation.z : 0;
    const squashAmount = typeof options.squashAmount === 'number' ? options.squashAmount : 0.15;
    const rotationAmount = typeof options.rotationAmount === 'number' ? options.rotationAmount : 0.2;

    const state = { t: 0 };
    const tween = new tweenNs.Tween(state)
        .to({ t: 1 }, duration)
        .easing(options.easing || tweenNs.Easing.Cubic.InOut)
        .onStart(() => {
            if (typeof options.onStart === 'function') options.onStart();
        })
        .onUpdate(() => {
            const t = clamp01(state.t);
            const inv = 1 - t;
            const x = inv * inv * startVec.x + 2 * inv * t * control.x + t * t * endVec.x;
            const y = inv * inv * startVec.y + 2 * inv * t * control.y + t * t * endVec.y;
            const z = inv * inv * startVec.z + 2 * inv * t * control.z + t * t * endVec.z;
            object3d.position.set(x, y, z);

            const sinCurve = Math.sin(Math.PI * t);
            if (baseScale && squashAmount) {
                const stretch = 1 + squashAmount * sinCurve;
                const squash = 1 - squashAmount * 0.75 * sinCurve;
                object3d.scale.set(
                    baseScale.x * stretch,
                    baseScale.y * squash,
                    baseScale.z * stretch
                );
            }
            if (object3d.rotation && rotationAmount) {
                object3d.rotation.z = baseRotation + rotationAmount * sinCurve;
            }
            if (typeof options.onProgress === 'function') {
                options.onProgress(t);
            }
        })
        .onComplete(() => {
            object3d.position.copy(endVec);
            if (baseScale) object3d.scale.copy(baseScale);
            if (object3d.rotation) object3d.rotation.z = baseRotation;
            if (typeof options.onComplete === 'function') options.onComplete();
        });

    if (typeof options.delay === 'number') {
        tween.delay(Math.max(0, options.delay));
    }

    return tween;
}

/**
 * Creates a two-stage bounce tween (overshoot + settle) for vertical motion.
 * @param {THREE.Object3D} object3d
 * @param {object} options
 * @returns {TWEEN.Tween|null}
 */
export function createBounceTween(object3d, options = {}) {
    const tweenNs = getTween();
    if (!tweenNs || !object3d || !object3d.position) return null;

    const axis = options.axis || 'y';
    const baseScale = object3d.scale ? object3d.scale.clone() : null;
    const baseRotation = object3d.rotation ? object3d.rotation.z : 0;

    const startValue = options.start != null ? options.start : object3d.position[axis];
    const endValue = options.end != null ? options.end : object3d.position[axis];
    const distance = endValue - startValue;
    const direction = distance >= 0 ? 1 : -1;
    const overshootMag = options.overshoot != null
        ? Math.abs(options.overshoot)
        : Math.max(10, Math.abs(distance) * 0.25);
    const overshootValue = endValue + direction * overshootMag;

    const squashAmount = typeof options.squashAmount === 'number' ? options.squashAmount : 0.18;
    const rotationAmount = typeof options.rotationAmount === 'number' ? options.rotationAmount : 0.15;

    const state = { value: startValue };
    const applyUpdate = () => {
        object3d.position[axis] = state.value;
        const travelled = state.value - startValue;
        const travelTotal = overshootValue - startValue || 1;
        const rawProgress = travelled / travelTotal;
        const clampProgress = clamp01(rawProgress);
        const sine = Math.sin(Math.PI * clampProgress);
        if (baseScale && squashAmount) {
            const stretch = 1 + squashAmount * Math.abs(sine);
            const squash = 1 - squashAmount * 0.8 * Math.abs(sine);
            object3d.scale.set(
                baseScale.x * stretch,
                baseScale.y * squash,
                baseScale.z * stretch
            );
        }
        if (object3d.rotation && rotationAmount) {
            object3d.rotation.z = baseRotation + rotationAmount * sine * direction;
        }
        if (typeof options.onProgress === 'function') {
            const normalised = distance === 0 ? 1 : (state.value - startValue) / distance;
            options.onProgress(normalised, state.value);
        }
        if (typeof options.onUpdate === 'function') {
            options.onUpdate(state.value);
        }
    };

    const riseDuration = Math.max(50, options.duration || 600);
    const settleDuration = Math.max(50, options.settleDuration || 350);

    const firstTween = new tweenNs.Tween(state)
        .to({ value: overshootValue }, riseDuration)
        .easing(options.easingOut || tweenNs.Easing.Cubic.Out)
        .onStart(() => {
            if (typeof options.onStart === 'function') options.onStart();
        })
        .onUpdate(applyUpdate);

    if (typeof options.delay === 'number') {
        firstTween.delay(Math.max(0, options.delay));
    }

    const settleTween = new tweenNs.Tween(state)
        .to({ value: endValue }, settleDuration)
        .easing(options.settleEasing || tweenNs.Easing.Bounce.Out)
        .onUpdate(applyUpdate)
        .onComplete(() => {
            object3d.position[axis] = endValue;
            if (baseScale) object3d.scale.copy(baseScale);
            if (object3d.rotation) object3d.rotation.z = baseRotation;
            if (typeof options.onComplete === 'function') options.onComplete();
        });

    firstTween.chain(settleTween);
    return firstTween;
}

/**
 * Produces a looping float tween (yoyo) for gentle idle motion.
 * @param {THREE.Object3D} object3d
 * @param {object} options
 * @returns {TWEEN.Tween|null}
 */
export function createLoopingFloatTween(object3d, options = {}) {
    const tweenNs = getTween();
    if (!tweenNs || !object3d || !object3d.position) return null;

    const axis = options.axis || 'y';
    const amplitude = options.amplitude != null ? options.amplitude : 6;
    const duration = Math.max(200, options.duration || 1600);
    const baseValue = options.base != null ? options.base : object3d.position[axis];

    const state = { value: 0 };
    const tween = new tweenNs.Tween(state)
        .to({ value: 1 }, duration)
        .easing(tweenNs.Easing.Sine.InOut)
        .yoyo(true)
        .repeat(Infinity)
        .onUpdate(() => {
            const progress = clamp01(state.value);
            object3d.position[axis] = baseValue + Math.sin(progress * Math.PI) * amplitude;
        })
        .onStop(() => {
            object3d.position[axis] = baseValue;
        });

    return tween;
}
