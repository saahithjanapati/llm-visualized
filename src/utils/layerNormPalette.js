import * as THREE from 'three';

const DEFAULT_FINAL_COLOR = 0xf2ede4;
const DEFAULT_DARK_TARGET = new THREE.Color(0x1f1f1f);
const DEFAULT_WARM_HIGHLIGHT = new THREE.Color(0xfff4cc);
const DEFAULT_COOL_HIGHLIGHT = new THREE.Color(0xd4e4ff);

function resolveFinalColor(input) {
    if (input instanceof THREE.Color) {
        return input.clone();
    }
    const resolved = new THREE.Color();
    if (typeof input === 'number') {
        resolved.setHex(input);
    } else if (typeof input === 'string') {
        resolved.setStyle(input);
    } else {
        resolved.setHex(DEFAULT_FINAL_COLOR);
    }
    return resolved;
}

/**
 * Build a palette of colours used during the LayerNorm activation animation.
 * The palette keeps the final colour configurable while generating sensible
 * transition tones so that the animation retains depth even when the final
 * colour is bright (e.g. white).
 *
 * @param {number|string|THREE.Color} [finalColorInput]
 * @returns {{ final: THREE.Color, dark: THREE.Color, mid: THREE.Color, bright: THREE.Color }}
 */
export function buildLayerNormPalette(finalColorInput) {
    const finalColor = resolveFinalColor(finalColorInput);

    const hsl = { h: 0, s: 0, l: 0 };
    finalColor.getHSL(hsl);

    const highlightBlendTarget = DEFAULT_WARM_HIGHLIGHT.clone().lerp(DEFAULT_COOL_HIGHLIGHT, THREE.MathUtils.clamp((hsl.h - 0.45) * 4, 0, 1));

    const darkMix = THREE.MathUtils.clamp(0.5 + 0.3 * hsl.l, 0.35, 0.85);
    const midMix = THREE.MathUtils.clamp(0.55 + 0.25 * (1 - hsl.l), 0.4, 0.85);

    const dark = finalColor.clone().lerp(DEFAULT_DARK_TARGET, darkMix);
    const mid = finalColor.clone().lerp(highlightBlendTarget, midMix);

    return {
        final: finalColor.clone(),
        dark,
        mid,
        bright: finalColor.clone()
    };
}

