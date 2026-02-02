const easeInOutSine = (t) => 0.5 - 0.5 * Math.cos(Math.PI * t);
const easeOutSine = (t) => Math.sin((t * Math.PI) / 2);
const easeInSine = (t) => 1 - Math.cos((t * Math.PI) / 2);

export { easeInOutSine, easeOutSine, easeInSine };

let joyfulPatched = false;

export function applyJoyfulTweenEasings() {
    if (typeof TWEEN === 'undefined' || !TWEEN?.Easing) return false;
    if (joyfulPatched || TWEEN.__joyfulPatched) return true;

    const existingEasing = TWEEN.Easing;
    const sinusoidal = existingEasing.Sinusoidal || {
        In: easeInSine,
        Out: easeOutSine,
        InOut: easeInOutSine
    };

    let patched = false;
    const quadratic = existingEasing.Quadratic;
    if (quadratic) {
        const canMutateQuadratic = !Object.isFrozen(quadratic) && ['In', 'Out', 'InOut'].every((key) => {
            const desc = Object.getOwnPropertyDescriptor(quadratic, key);
            return !desc || !!desc.writable;
        });

        if (canMutateQuadratic) {
            quadratic.In = sinusoidal.In;
            quadratic.Out = sinusoidal.Out;
            quadratic.InOut = sinusoidal.InOut;
            patched = true;
        }
    }

    if (!patched) {
        const easingDesc = Object.getOwnPropertyDescriptor(TWEEN, 'Easing');
        const canReplaceEasing = !easingDesc || !!easingDesc.writable || typeof easingDesc.set === 'function';
        if (canReplaceEasing) {
            try {
                const replacement = { ...existingEasing };
                if (!replacement.Sinusoidal) replacement.Sinusoidal = sinusoidal;
                replacement.Quadratic = {
                    In: sinusoidal.In,
                    Out: sinusoidal.Out,
                    InOut: sinusoidal.InOut
                };
                TWEEN.Easing = replacement;
                patched = true;
            } catch (err) {
                patched = false;
            }
        }
    }

    joyfulPatched = patched;
    if (patched) {
        try {
            TWEEN.__joyfulPatched = true;
        } catch (err) {
            // Ignore if TWEEN is sealed/frozen.
        }
    }
    return patched;
}
