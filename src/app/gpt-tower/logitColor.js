function hashStringToSeed(value) {
    if (!value) return 0;
    let hash = 0;
    for (let i = 0; i < value.length; i += 1) {
        hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0;
    }
    return hash >>> 0;
}

function hashToUnit(seed) {
    let x = seed >>> 0;
    x ^= x >>> 16;
    x = Math.imul(x, 0x7feb352d);
    x ^= x >>> 15;
    x = Math.imul(x, 0x846ca68b);
    x ^= x >>> 16;
    return (x >>> 0) / 4294967295;
}

export function resolveLogitTokenSeed(entry, fallbackIndex = 0) {
    if (entry && Number.isFinite(entry.token_id)) {
        return Math.floor(entry.token_id) >>> 0;
    }
    if (entry && typeof entry.token === 'string') {
        return hashStringToSeed(entry.token);
    }
    return (fallbackIndex ?? 0) >>> 0;
}

export function getLogitTokenColorUnit(seed) {
    const safeSeed = (Number.isFinite(seed) ? Math.floor(seed) : 0) >>> 0;
    const h = hashToUnit(safeSeed);
    const s = 0.78 + 0.18 * hashToUnit(safeSeed ^ 0x9e3779b9);
    const l = 0.5 + 0.18 * hashToUnit(safeSeed ^ 0x85ebca6b);
    return { h, s, l };
}

export function getLogitTokenColorCss(seed, alpha = 1) {
    const { h, s, l } = getLogitTokenColorUnit(seed);
    const hue = (h * 360).toFixed(2);
    const sat = (s * 100).toFixed(2);
    const light = (l * 100).toFixed(2);
    const opacity = Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 1;
    return `hsl(${hue} ${sat}% ${light}% / ${opacity.toFixed(3)})`;
}

