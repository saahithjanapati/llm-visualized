function hashStringToSeed(value) {
    if (!value) return 0;
    let hash = 0;
    for (let i = 0; i < value.length; i += 1) {
        hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0;
    }
    return hash >>> 0;
}

function mixSeed(seed) {
    let x = seed >>> 0;
    x ^= x >>> 16;
    x = Math.imul(x, 0x7feb352d);
    x ^= x >>> 15;
    x = Math.imul(x, 0x846ca68b);
    x ^= x >>> 16;
    return x >>> 0;
}

function hashToUnit(seed) {
    return mixSeed(seed) / 4294967295;
}

const ADJACENT_TOKEN_MIN_HUE_DISTANCE = 0.11;
const ADJACENT_TOKEN_COLOR_VARIANTS = Object.freeze([
    0x00000000,
    0x9e3779b9,
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
    0x165667b1,
    0xd1b54a35
]);
// Fixed neon palette for prompt chips so token colors stay stable across devices.
const LOGIT_TOKEN_CHIP_PALETTE = Object.freeze([
    Object.freeze([255, 88, 171]),
    Object.freeze([255, 130, 64]),
    Object.freeze([255, 214, 74]),
    Object.freeze([193, 255, 71]),
    Object.freeze([81, 255, 140]),
    Object.freeze([46, 244, 255]),
    Object.freeze([60, 171, 255]),
    Object.freeze([124, 119, 255]),
    Object.freeze([201, 101, 255]),
    Object.freeze([255, 95, 126]),
    Object.freeze([255, 111, 214]),
    Object.freeze([102, 255, 221])
]);

export function resolveLogitTokenSeed(entry, fallbackIndex = 0) {
    if (entry && Number.isFinite(entry.token_id)) {
        return Math.floor(entry.token_id) >>> 0;
    }
    if (entry && typeof entry.token === 'string') {
        return hashStringToSeed(entry.token);
    }
    return (fallbackIndex ?? 0) >>> 0;
}

export function resolveLogitTokenEntrySeed(entry, fallbackIndex = 0) {
    if (!entry || typeof entry !== 'object') return resolveLogitTokenSeed(null, fallbackIndex);
    if (Number.isFinite(entry.seed)) return Math.floor(entry.seed) >>> 0;
    return resolveLogitTokenSeed({
        token_id: entry.tokenId ?? entry.token_id,
        token: entry.tokenLabel ?? entry.token
    }, fallbackIndex);
}

function resolveLogitTokenChipSeed(entry, fallbackIndex = 0) {
    if (!entry || typeof entry !== 'object') return resolveLogitTokenSeed(null, fallbackIndex);

    const rawTokenId = Number(entry.tokenId ?? entry.token_id);
    if (Number.isFinite(rawTokenId)) {
        return Math.floor(rawTokenId) >>> 0;
    }

    const tokenLabel = entry.tokenLabel ?? entry.token;
    if (typeof tokenLabel === 'string' && tokenLabel.length) {
        return hashStringToSeed(tokenLabel);
    }

    if (Number.isFinite(entry.seed)) {
        return Math.floor(entry.seed) >>> 0;
    }

    return resolveLogitTokenSeed(null, fallbackIndex);
}

function normalizeLogitTokenChipColorKey(colorKey, paletteSize) {
    if (!(paletteSize > 0)) return 0;
    return ((Number.isFinite(colorKey) ? Math.floor(colorKey) : 0) % paletteSize + paletteSize) % paletteSize;
}

export function getLogitTokenColorUnit(seed) {
    const safeSeed = (Number.isFinite(seed) ? Math.floor(seed) : 0) >>> 0;
    const h = hashToUnit(safeSeed);
    const s = 0.78 + 0.18 * hashToUnit(safeSeed ^ 0x9e3779b9);
    const l = 0.5 + 0.18 * hashToUnit(safeSeed ^ 0x85ebca6b);
    return { h, s, l };
}

function circularUnitDistance(a, b) {
    const raw = Math.abs(a - b) % 1;
    return raw > 0.5 ? 1 - raw : raw;
}

function resolveEntryIdentity(entry) {
    if (!entry || typeof entry !== 'object') {
        return { tokenId: null, tokenLabel: '' };
    }
    const rawTokenId = Number(entry.tokenId ?? entry.token_id);
    const tokenId = Number.isFinite(rawTokenId) ? Math.floor(rawTokenId) : null;
    const tokenLabel = entry.tokenLabel ?? entry.token ?? '';
    return {
        tokenId,
        tokenLabel: String(tokenLabel)
    };
}

function canAdjacentEntriesShareColor(leftEntry, rightEntry) {
    const left = resolveEntryIdentity(leftEntry);
    const right = resolveEntryIdentity(rightEntry);
    if (Number.isFinite(left.tokenId) && Number.isFinite(right.tokenId)) {
        return left.tokenId === right.tokenId;
    }
    return left.tokenLabel.length > 0
        && right.tokenLabel.length > 0
        && left.tokenLabel === right.tokenLabel;
}

export function resolveLogitTokenChipColorKey(entry, fallbackIndex = 0) {
    const paletteSize = LOGIT_TOKEN_CHIP_PALETTE.length;
    if (!paletteSize) return 0;
    const seed = resolveLogitTokenChipSeed(entry, fallbackIndex);
    return mixSeed(seed) % paletteSize;
}

function resolveAlternateChipColorKey(baseColorKey, entrySeed, entryIndex, paletteSize) {
    if (paletteSize <= 1) return baseColorKey;
    const rotationSeed = mixSeed(
        (entrySeed ^ Math.imul(((entryIndex ?? 0) + 1) >>> 0, 0x9e3779b9)) >>> 0
    );
    const rotationStart = rotationSeed % (paletteSize - 1);
    return (baseColorKey + 1 + rotationStart) % paletteSize;
}

export function resolveAdjacentLogitTokenChipColorKeys(entries = [], { preferredColorKeys = null } = {}) {
    const safeEntries = Array.isArray(entries) ? entries : [];
    const paletteSize = LOGIT_TOKEN_CHIP_PALETTE.length;
    if (!paletteSize) {
        return safeEntries.map(() => 0);
    }

    const resolvedColorKeys = [];
    let previousEntry = null;
    let previousColorKey = null;

    safeEntries.forEach((entry, index) => {
        const preferredColorKey = Array.isArray(preferredColorKeys) ? preferredColorKeys[index] : null;
        const baseColorKey = Number.isFinite(preferredColorKey)
            ? normalizeLogitTokenChipColorKey(preferredColorKey, paletteSize)
            : resolveLogitTokenChipColorKey(entry, index);
        const entrySeed = resolveLogitTokenChipSeed(entry, index);
        const chosenColorKey = (
            previousColorKey === null
            || canAdjacentEntriesShareColor(previousEntry, entry)
            || baseColorKey !== previousColorKey
        )
            ? baseColorKey
            : resolveAlternateChipColorKey(baseColorKey, entrySeed, index, paletteSize);

        resolvedColorKeys.push(chosenColorKey);
        previousEntry = entry;
        previousColorKey = chosenColorKey;
    });

    return resolvedColorKeys;
}

export function resolveAdjacentLogitTokenSeeds(entries = []) {
    const safeEntries = Array.isArray(entries) ? entries : [];
    const resolvedSeeds = [];
    let previousEntry = null;
    let previousSeed = null;

    safeEntries.forEach((entry, index) => {
        const baseSeed = resolveLogitTokenEntrySeed(entry, index);
        if (previousSeed == null || canAdjacentEntriesShareColor(previousEntry, entry)) {
            resolvedSeeds.push(baseSeed);
            previousEntry = entry;
            previousSeed = baseSeed;
            return;
        }

        const previousHue = getLogitTokenColorUnit(previousSeed).h;
        let chosenSeed = baseSeed;
        let bestHueDistance = circularUnitDistance(getLogitTokenColorUnit(baseSeed).h, previousHue);

        if (bestHueDistance < ADJACENT_TOKEN_MIN_HUE_DISTANCE) {
            const positionSalt = Math.imul((index + 1) >>> 0, 0x45d9f3b) >>> 0;
            ADJACENT_TOKEN_COLOR_VARIANTS.forEach((variant) => {
                const candidateSeed = variant === 0
                    ? baseSeed
                    : ((baseSeed ^ variant) + positionSalt) >>> 0;
                const hueDistance = circularUnitDistance(getLogitTokenColorUnit(candidateSeed).h, previousHue);
                if (hueDistance > bestHueDistance) {
                    bestHueDistance = hueDistance;
                    chosenSeed = candidateSeed;
                }
            });
        }

        resolvedSeeds.push(chosenSeed);
        previousEntry = entry;
        previousSeed = chosenSeed;
    });

    return resolvedSeeds;
}

export function getLogitTokenColorCss(seed, alpha = 1) {
    const { h, s, l } = getLogitTokenColorUnit(seed);
    const hue = (h * 360).toFixed(2);
    const sat = (s * 100).toFixed(2);
    const light = (l * 100).toFixed(2);
    const opacity = Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 1;
    return `hsl(${hue} ${sat}% ${light}% / ${opacity.toFixed(3)})`;
}

export function getLogitTokenChipColorCss(colorKey, alpha = 1) {
    const paletteSize = LOGIT_TOKEN_CHIP_PALETTE.length;
    const safeKey = normalizeLogitTokenChipColorKey(colorKey, paletteSize);
    const [red, green, blue] = LOGIT_TOKEN_CHIP_PALETTE[safeKey] || [255, 255, 255];
    const opacity = Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 1;
    return `rgb(${red} ${green} ${blue} / ${opacity.toFixed(3)})`;
}

export function getLogitTokenChipColorHex(colorKey) {
    const paletteSize = LOGIT_TOKEN_CHIP_PALETTE.length;
    const safeKey = normalizeLogitTokenChipColorKey(colorKey, paletteSize);
    const [red, green, blue] = LOGIT_TOKEN_CHIP_PALETTE[safeKey] || [255, 255, 255];
    return (red << 16) | (green << 8) | blue;
}
