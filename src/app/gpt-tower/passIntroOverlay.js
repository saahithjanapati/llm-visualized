import * as THREE from 'three';
import {
    LAYER_NORM_1_Y_POS,
    LN_PARAMS,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM,
    EMBEDDING_BOTTOM_Y_ADJUST,
    EMBEDDING_BOTTOM_VOCAB_X_OFFSET,
    VECTOR_DEPTH_SPACING_BASE,
    MAX_LANE_DEPTH_RATIO,
    MIN_LANE_DEPTH
} from '../../utils/constants.js';
import { TOKEN_CHIP_STYLE } from './config.js';

const FIRST_PASS_TYPE_MS = 42;
const NEXT_PASS_TYPE_MS = 28;
const TYPE_SETTLE_MS = 180;
const TOKENIZE_IN_PLACE_DURATION_MS = 360;
const TOKENIZE_STAGGER_MS = 20;
const TOKENIZE_HOLD_MS = 220;
const HANDOFF_BASE_DURATION_MS = 760;
const HANDOFF_STAGGER_MS = 26;
const HANDOFF_MIN_ARC_PX = 56;
const HANDOFF_MAX_ARC_PX = 190;
const HANDOFF_COMMIT_PROGRESS = 0.985;
const HANDOFF_FADE_START_PROGRESS = 0.93;

const TOKEN_REPLACEMENTS = new Map([
    ['Âł', ' '],
    ['âĢ¦', '...'],
    ['âĢĵ', "'"],
    ['âĢĶ', '"']
]);

function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function easeOutCubic(t) {
    const x = clamp(t, 0, 1);
    return 1 - ((1 - x) ** 3);
}

function easeInCubic(t) {
    const x = clamp(t, 0, 1);
    return x ** 3;
}

function easeInOutCubic(t) {
    const x = clamp(t, 0, 1);
    if (x < 0.5) return 4 * x * x * x;
    return 1 - ((-2 * x + 2) ** 3) / 2;
}

function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));
}

function commonPrefixLength(a, b) {
    const maxLen = Math.min(a.length, b.length);
    let i = 0;
    while (i < maxLen && a.charCodeAt(i) === b.charCodeAt(i)) i += 1;
    return i;
}

function decodeTokenToRawText(token) {
    if (token === null || token === undefined) return '';
    const raw = String(token);
    if (!raw) return '';
    if (TOKEN_REPLACEMENTS.has(raw)) {
        return TOKEN_REPLACEMENTS.get(raw);
    }

    return raw
        .replace(/\u0120/g, ' ')
        .replace(/\u010A/g, '\n')
        .replace(/\u0109/g, '\t')
        .replace(/\u00A0/g, ' ');
}

function resolveLaneDepth(laneCount) {
    const safeLaneCount = Math.max(1, Math.floor(laneCount || 1));
    const desiredDepth = (safeLaneCount + 1) * VECTOR_DEPTH_SPACING_BASE;
    const cappedDepthRaw = desiredDepth * MAX_LANE_DEPTH_RATIO;
    return Math.max(MIN_LANE_DEPTH, cappedDepthRaw);
}

function buildOverlayDom() {
    let root = document.getElementById('passIntroOverlay');
    if (!root) {
        root = document.createElement('div');
        root.id = 'passIntroOverlay';
        root.dataset.visible = 'false';
        root.innerHTML = `
            <div class="pass-intro-scrim"></div>
            <div class="pass-intro-stage">
                <div class="pass-intro-window" data-role="window">
                    <div class="pass-intro-window-header">
                        <span class="pass-intro-dot pass-intro-dot--red" aria-hidden="true"></span>
                        <span class="pass-intro-dot pass-intro-dot--amber" aria-hidden="true"></span>
                        <span class="pass-intro-dot pass-intro-dot--green" aria-hidden="true"></span>
                        <span class="pass-intro-window-title">Prompt</span>
                    </div>
                    <div class="pass-intro-editor">
                        <div class="pass-intro-text" data-role="text"></div>
                        <span class="pass-intro-cursor" data-role="cursor">|</span>
                    </div>
                </div>
                <div class="pass-intro-token-layer" data-role="token-layer"></div>
            </div>
        `;
        document.body.appendChild(root);
    }

    return {
        root,
        scrimEl: root.querySelector('.pass-intro-scrim'),
        stage: root.querySelector('.pass-intro-stage'),
        windowEl: root.querySelector('[data-role="window"]'),
        textEl: root.querySelector('[data-role="text"]'),
        cursorEl: root.querySelector('[data-role="cursor"]'),
        tokenLayer: root.querySelector('[data-role="token-layer"]')
    };
}

function resolveLandingPoint({
    pipeline,
    laneCount,
    laneLayoutIndex,
    rendererRect
}) {
    const safeLaneCount = Math.max(1, Math.floor(laneCount || 1));
    const safeLayoutIndex = Math.max(0, Math.floor(Number.isFinite(laneLayoutIndex) ? laneLayoutIndex : 0));

    const laneDepth = resolveLaneDepth(safeLaneCount);
    const laneSpacing = laneDepth / (safeLaneCount + 1);
    const laneZ = -laneDepth / 2 + laneSpacing * (safeLayoutIndex + 1);

    const residualYBase = LAYER_NORM_1_Y_POS
        - LN_PARAMS.height / 2
        + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM;
    const bottomVocabCenterY = residualYBase
        - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2
        + EMBEDDING_BOTTOM_Y_ADJUST;
    const vocabBottomY = bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2;

    const chipHeight = (Number.isFinite(TOKEN_CHIP_STYLE.height) ? TOKEN_CHIP_STYLE.height : TOKEN_CHIP_STYLE.minHeight)
        * (Number.isFinite(TOKEN_CHIP_STYLE.scale) && TOKEN_CHIP_STYLE.scale > 0 ? TOKEN_CHIP_STYLE.scale : 1);
    const staticY = vocabBottomY - chipHeight / 2 - TOKEN_CHIP_STYLE.staticGap;
    const staticZ = laneZ + (Number.isFinite(TOKEN_CHIP_STYLE.staticZOffset) ? TOKEN_CHIP_STYLE.staticZOffset : 0);

    const worldPoint = new THREE.Vector3(EMBEDDING_BOTTOM_VOCAB_X_OFFSET, staticY, staticZ);

    const camera = pipeline?.engine?.camera;
    if (!camera || !rendererRect) {
        return {
            x: window.innerWidth * 0.5,
            y: window.innerHeight * 0.78
        };
    }

    const projected = worldPoint.project(camera);
    if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y)) {
        return {
            x: window.innerWidth * 0.5,
            y: window.innerHeight * 0.78
        };
    }

    return {
        x: rendererRect.left + ((projected.x + 1) * 0.5) * rendererRect.width,
        y: rendererRect.top + ((1 - projected.y) * 0.5) * rendererRect.height
    };
}

function buildTokenEntries({
    activationSource,
    laneCount,
    laneTokenIndices,
    laneLayoutIndices,
    tokenLabels
}) {
    const safeLaneCount = Math.max(1, Math.floor(laneCount || 1));
    const entries = [];

    for (let lanePos = 0; lanePos < safeLaneCount; lanePos += 1) {
        const tokenIndex = Array.isArray(laneTokenIndices) && Number.isFinite(laneTokenIndices[lanePos])
            ? Math.floor(laneTokenIndices[lanePos])
            : lanePos;
        const layoutIndex = Array.isArray(laneLayoutIndices) && Number.isFinite(laneLayoutIndices[lanePos])
            ? Math.floor(laneLayoutIndices[lanePos])
            : lanePos;
        const rawToken = Array.isArray(tokenLabels) && typeof tokenLabels[lanePos] === 'string'
            ? tokenLabels[lanePos]
            : (activationSource && typeof activationSource.getTokenString === 'function'
                ? activationSource.getTokenString(tokenIndex)
                : '');
        const tokenId = (activationSource && typeof activationSource.getTokenId === 'function')
            ? activationSource.getTokenId(tokenIndex)
            : null;

        entries.push({
            lanePos,
            layoutIndex,
            tokenIndex,
            tokenId,
            rawToken: rawToken ?? '',
            rawText: decodeTokenToRawText(rawToken ?? '')
        });
    }

    return entries;
}

function normalizeInlineTokenText(value) {
    if (typeof value !== 'string' || !value.length) return ' ';
    return value.replace(/ /g, '\u00A0');
}

function createInlineTokenElement(entry, index) {
    const token = document.createElement('span');
    token.className = 'pass-intro-inline-token';
    token.style.setProperty('--chip-hue', String((index * 41) % 360));
    token.style.setProperty('--tokenize-delay-ms', `${index * TOKENIZE_STAGGER_MS}ms`);

    const label = document.createElement('span');
    label.className = 'pass-intro-inline-token-label';
    const displayText = normalizeInlineTokenText(entry.rawText);
    label.textContent = displayText;

    if (!entry.rawText || /^\s+$/.test(entry.rawText)) {
        token.dataset.whitespace = 'true';
    }

    const idBadge = document.createElement('span');
    idBadge.className = 'pass-intro-inline-token-id';
    idBadge.textContent = Number.isFinite(entry.tokenId) ? `#${Math.floor(entry.tokenId)}` : '#?';

    token.append(label, idBadge);
    return token;
}

export function initPassIntroOverlay({ pipeline, activationSource } = {}) {
    if (typeof document === 'undefined' || typeof window === 'undefined') {
        return {
            play: async () => {},
            dispose: () => {}
        };
    }

    const dom = buildOverlayDom();
    let disposed = false;
    let hasPlayedOnce = false;
    let currentRawText = '';
    let activeHandoffRaf = null;

    const clearHandoffInlineState = () => {
        dom.windowEl.style.opacity = '';
        dom.windowEl.style.transform = '';
        if (dom.scrimEl) dom.scrimEl.style.opacity = '';
    };

    const hideOverlay = () => {
        if (activeHandoffRaf) {
            cancelAnimationFrame(activeHandoffRaf);
            activeHandoffRaf = null;
        }
        dom.root.dataset.visible = 'false';
        dom.root.classList.remove('is-tokenized', 'is-handoff');
        dom.windowEl.classList.remove('is-transitioning');
        dom.textEl.classList.remove('is-tokenized', 'is-faded');
        clearHandoffInlineState();
        dom.tokenLayer.innerHTML = '';
        document.body.classList.remove('pass-intro-active');
    };

    const animateHandoffToTower = ({
        chips,
        entries,
        laneCount,
        rendererRect,
        onCommit = null
    }) => {
        if (!Array.isArray(chips) || !chips.length) return Promise.resolve();

        let commitFired = false;
        const trajectories = chips.map((chip, idx) => {
            const startX = Number.parseFloat(chip.style.left) || 0;
            const startY = Number.parseFloat(chip.style.top) || 0;
            const landing = resolveLandingPoint({
                pipeline,
                laneCount,
                laneLayoutIndex: entries[idx]?.layoutIndex ?? idx,
                rendererRect
            });
            const dx = landing.x - startX;
            const dy = landing.y - startY;
            const distance = Math.hypot(dx, dy);
            const laneSkew = (idx / Math.max(1, chips.length - 1)) - 0.5;
            const baseArc = clamp(distance * 0.16, HANDOFF_MIN_ARC_PX, HANDOFF_MAX_ARC_PX);

            return {
                chip,
                startX,
                startY,
                targetX: landing.x,
                targetY: landing.y,
                delayMs: idx * HANDOFF_STAGGER_MS,
                durationMs: HANDOFF_BASE_DURATION_MS + clamp(distance * 0.1, 0, 180),
                arcPx: baseArc + laneSkew * 22
            };
        });

        const totalMs = trajectories.reduce(
            (max, entry) => Math.max(max, entry.delayMs + entry.durationMs),
            0
        );

        return new Promise((resolve) => {
            const startMs = performance.now();
            const tick = (now) => {
                if (disposed) {
                    activeHandoffRaf = null;
                    resolve();
                    return;
                }

                const elapsedMs = now - startMs;
                const globalProgress = clamp(elapsedMs / Math.max(1, totalMs), 0, 1);
                const windowProgress = easeInOutCubic(globalProgress);
                dom.windowEl.style.opacity = String(lerp(0.26, 0.06, windowProgress));
                dom.windowEl.style.transform = `translateX(-50%) translateY(${lerp(-6, -26, windowProgress).toFixed(2)}px) scale(${lerp(1, 0.94, windowProgress).toFixed(4)})`;
                if (dom.scrimEl) {
                    dom.scrimEl.style.opacity = String(lerp(0.4, 0.08, windowProgress));
                }
                if (!commitFired && globalProgress >= HANDOFF_COMMIT_PROGRESS) {
                    commitFired = true;
                    if (typeof onCommit === 'function') {
                        try { onCommit(); } catch (_) { /* no-op */ }
                    }
                }

                let allDone = true;
                trajectories.forEach((entry) => {
                    const localElapsedMs = elapsedMs - entry.delayMs;
                    const localRaw = clamp(localElapsedMs / Math.max(1, entry.durationMs), 0, 1);
                    if (localRaw < 1) allDone = false;
                    if (localRaw <= 0) return;

                    const p = easeOutCubic(localRaw);
                    const x = lerp(entry.startX, entry.targetX, p);
                    const arch = Math.sin(Math.PI * p) * entry.arcPx;
                    const y = lerp(entry.startY, entry.targetY, p) - arch;
                    const scale = lerp(1, 0.36, p);
                    const rotation = lerp(0, (entry.targetX - entry.startX) * 0.012, 1 - p);

                    let opacity = 1;
                    if (localRaw > HANDOFF_FADE_START_PROGRESS) {
                        const fadeProgress = (localRaw - HANDOFF_FADE_START_PROGRESS)
                            / (1 - HANDOFF_FADE_START_PROGRESS);
                        // Keep chips visible until almost complete so they read as the same objects
                        // right up to the 3D handoff frame.
                        opacity = 1 - (easeInCubic(fadeProgress) * 0.92);
                    }

                    const blurPx = localRaw > 0.78
                        ? lerp(0, 1.1, (localRaw - 0.78) / 0.22)
                        : 0;

                    entry.chip.style.left = `${x.toFixed(2)}px`;
                    entry.chip.style.top = `${y.toFixed(2)}px`;
                    entry.chip.style.transform = `translate(-50%, -50%) scale(${scale.toFixed(4)}) rotate(${rotation.toFixed(2)}deg)`;
                    entry.chip.style.opacity = `${clamp(opacity, 0, 1).toFixed(4)}`;
                    entry.chip.style.filter = blurPx > 0.01 ? `blur(${blurPx.toFixed(2)}px)` : 'none';
                });

                if (allDone) {
                    activeHandoffRaf = null;
                    resolve();
                    return;
                }

                activeHandoffRaf = requestAnimationFrame(tick);
            };

            activeHandoffRaf = requestAnimationFrame(tick);
        });
    };

    const play = async ({
        laneCount,
        laneTokenIndices,
        laneLayoutIndices,
        tokenLabels,
        onHandoffCommit = null
    } = {}) => {
        if (disposed) return;

        const entries = buildTokenEntries({
            activationSource,
            laneCount,
            laneTokenIndices,
            laneLayoutIndices,
            tokenLabels
        });
        if (!entries.length) return;

        const nextRawText = entries.map((entry) => entry.rawText).join('');
        const normalizedNextText = String(nextRawText ?? '').replace(/\r/g, '').replace(/\u00A0/g, ' ');

        dom.root.dataset.visible = 'true';
        dom.root.classList.remove('is-tokenized', 'is-handoff');
        dom.windowEl.classList.remove('is-transitioning');
        dom.textEl.classList.remove('is-tokenized', 'is-faded');
        dom.tokenLayer.innerHTML = '';
        document.body.classList.add('pass-intro-active');

        if (disposed) return;

        const prefixLen = commonPrefixLength(currentRawText, normalizedNextText);
        let typedText = currentRawText;

        if (prefixLen < typedText.length) {
            typedText = typedText.slice(0, prefixLen);
            dom.textEl.textContent = typedText;
            await delay(120);
            if (disposed) return;
        } else {
            dom.textEl.textContent = typedText;
        }

        const suffix = normalizedNextText.slice(prefixLen);
        const typeDelay = hasPlayedOnce ? NEXT_PASS_TYPE_MS : FIRST_PASS_TYPE_MS;
        for (let i = 0; i < suffix.length; i += 1) {
            if (disposed) return;
            typedText += suffix[i];
            dom.textEl.textContent = typedText;
            await delay(typeDelay);
        }
        if (!suffix.length) {
            await delay(80);
        }

        currentRawText = normalizedNextText;
        dom.root.classList.add('is-tokenized');
        await delay(TYPE_SETTLE_MS);
        if (disposed) return;

        dom.textEl.textContent = '';
        dom.textEl.classList.add('is-tokenized');
        const inlineTokens = entries.map((entry, idx) => {
            const tokenEl = createInlineTokenElement(entry, idx);
            dom.textEl.appendChild(tokenEl);
            return tokenEl;
        });

        await nextFrame();
        if (disposed) return;

        inlineTokens.forEach((tokenEl) => {
            tokenEl.classList.add('is-visible');
        });

        const tokenizeTailMs = TOKENIZE_IN_PLACE_DURATION_MS
            + TOKENIZE_STAGGER_MS * Math.max(0, inlineTokens.length - 1);
        await delay(tokenizeTailMs + TOKENIZE_HOLD_MS);
        if (disposed) return;

        const chips = inlineTokens.map((tokenEl) => {
            const rect = tokenEl.getBoundingClientRect();
            const chip = tokenEl.cloneNode(true);
            chip.classList.remove('pass-intro-inline-token', 'is-visible');
            chip.classList.add('pass-intro-token-chip');
            chip.style.left = `${rect.left + rect.width / 2}px`;
            chip.style.top = `${rect.top + rect.height / 2}px`;
            chip.style.opacity = '1';
            chip.style.transform = 'translate(-50%, -50%) scale(1)';
            chip.style.transition = 'none';
            dom.tokenLayer.appendChild(chip);
            return chip;
        });

        dom.textEl.classList.add('is-faded');

        const rendererRect = pipeline?.engine?.renderer?.domElement?.getBoundingClientRect?.()
            || { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };

        dom.root.classList.add('is-handoff');
        dom.windowEl.classList.add('is-transitioning');

        await animateHandoffToTower({
            chips,
            entries,
            laneCount,
            rendererRect,
            onCommit: onHandoffCommit
        });
        await delay(40);
        if (disposed) return;

        hasPlayedOnce = true;
        hideOverlay();
    };

    const dispose = () => {
        if (disposed) return;
        disposed = true;
        hideOverlay();
    };

    return { play, dispose };
}

export default initPassIntroOverlay;
