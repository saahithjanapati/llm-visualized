import {
    applyTokenChipColors,
    buildPromptTokenChipEntries,
    resolvePromptTokenChipColorState
} from '../../ui/tokenChipColorUtils.js';
import { formatTokenLabel } from './tokenLabels.js';

const FIRST_PASS_TYPE_BASE_MS = 46;
const NEXT_PASS_TYPE_BASE_MS = 30;
const TYPE_DELETE_MS = 26;
const TYPE_SETTLE_MS = 320;
const TOKENIZE_IN_PLACE_DURATION_MS = 780;
const TOKENIZE_STAGGER_MS = 34;
const TOKENIZE_HOLD_MS = 1000;
const APPEND_TOKEN_STAGGER_MS = 110;
const APPEND_TOKEN_SETTLE_MS = 460;
const APPEND_TOKEN_HOLD_MS = 620;
const HANDOFF_BASE_DURATION_MS = 860;
const HANDOFF_STAGGER_MS = 28;
const HANDOFF_MIN_ARC_PX = 44;
const HANDOFF_MAX_ARC_PX = 160;
const OVERLAY_HIDE_CLEANUP_DELAY_MS = 220;

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

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (!overlay) return;
    overlay.classList.add('is-hidden');
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

function resolveTypingDelayMs(char, baseMs) {
    const safeBaseMs = Number.isFinite(baseMs) ? baseMs : FIRST_PASS_TYPE_BASE_MS;
    let delayMs = safeBaseMs * (0.9 + Math.random() * 0.9);

    if (/\s/.test(char)) {
        delayMs += 26 + Math.random() * 64;
    }
    if (/[,.!?;:]/.test(char)) {
        delayMs += 90 + Math.random() * 180;
    } else if (/[)"'\]]/.test(char)) {
        delayMs += 36 + Math.random() * 82;
    } else if (Math.random() < 0.16) {
        delayMs += 34 + Math.random() * 88;
    }

    return Math.round(delayMs);
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
                    </div>
                    <div class="pass-intro-editor" data-role="editor">
                        <div class="pass-intro-shell-line">
                            <div class="pass-intro-shell-content">
                                <div class="pass-intro-text" data-role="text"></div>
                            </div>
                        </div>
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
        windowEl: root.querySelector('[data-role="window"]'),
        editorEl: root.querySelector('[data-role="editor"]'),
        textEl: root.querySelector('[data-role="text"]'),
        tokenLayer: root.querySelector('[data-role="token-layer"]')
    };
}

function syncEditorScroll(editorEl) {
    if (!editorEl) return;
    editorEl.scrollTop = editorEl.scrollHeight;
}

function buildFallbackTargets({ rootRect = null, chipRects = [] } = {}) {
    const fallbackLeft = rootRect && rootRect.width > 0
        ? rootRect.left + 18
        : 18;
    const fallbackRight = rootRect && rootRect.width > 0
        ? rootRect.right - 18
        : Math.max(140, window.innerWidth - 18);
    const fallbackTop = rootRect && rootRect.height > 0
        ? rootRect.top + 14
        : Math.max(24, window.innerHeight - 74);
    const rowGap = 8;
    let cursorX = fallbackLeft;
    let cursorY = fallbackTop;

    return chipRects.map((rect) => {
        const safeWidth = Math.max(48, rect?.width || 48);
        const safeHeight = Math.max(24, rect?.height || 24);
        if (cursorX + safeWidth > fallbackRight && cursorX > fallbackLeft) {
            cursorX = fallbackLeft;
            cursorY += safeHeight + rowGap;
        }

        const target = {
            x: cursorX + safeWidth / 2,
            y: cursorY + safeHeight / 2,
            scale: 0.8
        };
        cursorX += safeWidth + rowGap;
        return target;
    });
}

function resolvePromptStripTargets({ promptTokenStrip, chipRects = [] } = {}) {
    const root = promptTokenStrip?.getRootElement?.()
        || document.getElementById('promptTokenStrip');
    const rootRect = root?.getBoundingClientRect?.() || null;
    const fallbackTargets = buildFallbackTargets({ rootRect, chipRects });

    return chipRects.map((chipRect, index) => {
        const tokenEl = promptTokenStrip?.getTokenElement?.(index)
            || root?.querySelector?.(`[data-token-entry-index="${index}"]`);
        const targetRect = tokenEl?.getBoundingClientRect?.() || null;
        if (!targetRect || targetRect.width <= 0 || targetRect.height <= 0) {
            return fallbackTargets[index];
        }

        const scaleX = targetRect.width / Math.max(1, chipRect?.width || 1);
        const scaleY = targetRect.height / Math.max(1, chipRect?.height || 1);
        return {
            x: targetRect.left + targetRect.width / 2,
            y: targetRect.top + targetRect.height / 2,
            scale: clamp(Math.min(scaleX, scaleY), 0.58, 1.08)
        };
    });
}

function buildTokenEntries({
    activationSource,
    laneCount,
    laneTokenIndices,
    tokenLabels
}) {
    const safeLaneCount = Math.max(1, Math.floor(laneCount || 1));
    const entries = [];

    for (let lanePos = 0; lanePos < safeLaneCount; lanePos += 1) {
        const tokenIndex = Array.isArray(laneTokenIndices) && Number.isFinite(laneTokenIndices[lanePos])
            ? Math.floor(laneTokenIndices[lanePos])
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
            tokenIndex,
            tokenId,
            rawToken: rawToken ?? '',
            rawText: decodeTokenToRawText(rawToken ?? ''),
            displayLabel: formatTokenLabel(rawToken ?? '')
        });
    }

    return entries;
}

function normalizeInlineTokenText(value) {
    if (typeof value !== 'string' || !value.length) return ' ';
    return value.replace(/ /g, '\u00A0');
}

function resolveTextNode(element) {
    if (!element) return null;
    for (let i = 0; i < element.childNodes.length; i += 1) {
        const node = element.childNodes[i];
        if (node && node.nodeType === Node.TEXT_NODE) {
            return node;
        }
    }
    return null;
}

function rectFromClientRectList(rectList) {
    if (!rectList || !rectList.length) return null;
    let left = Infinity;
    let top = Infinity;
    let right = -Infinity;
    let bottom = -Infinity;
    for (let i = 0; i < rectList.length; i += 1) {
        const rect = rectList[i];
        if (!rect) continue;
        left = Math.min(left, rect.left);
        top = Math.min(top, rect.top);
        right = Math.max(right, rect.right);
        bottom = Math.max(bottom, rect.bottom);
    }
    if (!Number.isFinite(left) || !Number.isFinite(top) || !Number.isFinite(right) || !Number.isFinite(bottom)) {
        return null;
    }
    return {
        left,
        top,
        right,
        bottom,
        width: Math.max(0, right - left),
        height: Math.max(0, bottom - top)
    };
}

function measureTokenTextRects(textEl, entries = []) {
    const textNode = resolveTextNode(textEl);
    if (!textNode) return [];

    const textContent = typeof textNode.textContent === 'string' ? textNode.textContent : '';
    let offset = 0;

    return entries.map((entry) => {
        const tokenText = typeof entry?.rawText === 'string' ? entry.rawText : '';
        const tokenLength = tokenText.length;
        const start = offset;
        const end = Math.min(textContent.length, start + tokenLength);
        offset = end;
        if (end <= start) return null;

        const range = document.createRange();
        range.setStart(textNode, start);
        range.setEnd(textNode, end);

        const boundingRect = range.getBoundingClientRect();
        if (boundingRect && boundingRect.width > 0 && boundingRect.height > 0) {
            return {
                left: boundingRect.left,
                top: boundingRect.top,
                right: boundingRect.right,
                bottom: boundingRect.bottom,
                width: boundingRect.width,
                height: boundingRect.height
            };
        }

        return rectFromClientRectList(range.getClientRects());
    });
}

function copyChipComputedStyle(sourceEl, chip) {
    if (!sourceEl || !chip || typeof window === 'undefined' || typeof window.getComputedStyle !== 'function') {
        return;
    }

    const computed = window.getComputedStyle(sourceEl);
    chip.style.display = computed.display;
    chip.style.alignItems = computed.alignItems;
    chip.style.justifyContent = computed.justifyContent;
    chip.style.boxSizing = computed.boxSizing;
    chip.style.fontFamily = computed.fontFamily;
    chip.style.fontSize = computed.fontSize;
    chip.style.fontWeight = computed.fontWeight;
    chip.style.letterSpacing = computed.letterSpacing;
    chip.style.lineHeight = computed.lineHeight;
    chip.style.whiteSpace = computed.whiteSpace;
    chip.style.color = computed.color;
    chip.style.paddingTop = computed.paddingTop;
    chip.style.paddingRight = computed.paddingRight;
    chip.style.paddingBottom = computed.paddingBottom;
    chip.style.paddingLeft = computed.paddingLeft;
    chip.style.borderTopWidth = computed.borderTopWidth;
    chip.style.borderRightWidth = computed.borderRightWidth;
    chip.style.borderBottomWidth = computed.borderBottomWidth;
    chip.style.borderLeftWidth = computed.borderLeftWidth;
    chip.style.borderTopStyle = computed.borderTopStyle;
    chip.style.borderRightStyle = computed.borderRightStyle;
    chip.style.borderBottomStyle = computed.borderBottomStyle;
    chip.style.borderLeftStyle = computed.borderLeftStyle;
    chip.style.borderTopColor = computed.borderTopColor;
    chip.style.borderRightColor = computed.borderRightColor;
    chip.style.borderBottomColor = computed.borderBottomColor;
    chip.style.borderLeftColor = computed.borderLeftColor;
    chip.style.borderRadius = computed.borderRadius;
    chip.style.minHeight = computed.minHeight;
    chip.style.background = computed.background;
    chip.style.boxShadow = computed.boxShadow;
}

function createInlinePromptToken({
    entry,
    index,
    colorState,
    promptTokenStrip,
    delayMs = index * TOKENIZE_STAGGER_MS
}) {
    const chip = document.createElement('span');
    chip.className = 'pass-intro-inline-token';
    chip.style.setProperty('--tokenize-delay-ms', `${Math.max(0, delayMs)}ms`);
    applyTokenChipColors(chip, entry?.chipEntry, index, { lookup: colorState?.lookup });
    const promptTokenEl = promptTokenStrip?.getTokenElement?.(index) || null;
    copyChipComputedStyle(promptTokenEl, chip);
    chip.style.display = 'inline-flex';
    chip.style.alignItems = 'center';
    chip.style.justifyContent = 'center';
    chip.style.transformOrigin = 'center center';

    const label = document.createElement('span');
    label.className = 'pass-intro-inline-token-label';
    label.textContent = normalizeInlineTokenText(entry?.displayLabel);
    label.style.display = 'block';
    label.style.lineHeight = '1';
    chip.appendChild(label);

    return chip;
}

function renderInlinePromptTokens(
    textEl,
    entries = [],
    colorState = null,
    promptTokenStrip = null,
    { delayResolver = null } = {}
) {
    if (!textEl) return [];

    const fragment = document.createDocumentFragment();
    const chipElements = [];
    entries.forEach((entry, index) => {
        const chip = createInlinePromptToken({
            entry,
            index,
            colorState,
            promptTokenStrip,
            delayMs: typeof delayResolver === 'function'
                ? delayResolver(index, entry)
                : (index * TOKENIZE_STAGGER_MS)
        });
        chipElements.push(chip);
        fragment.appendChild(chip);
    });

    textEl.replaceChildren(fragment);
    return chipElements;
}

function resolveAppendStartIndex(previousTokenCount, nextTokenCount) {
    if (!Number.isFinite(nextTokenCount) || nextTokenCount <= 0) return 0;
    if (!Number.isFinite(previousTokenCount) || previousTokenCount <= 0) return 0;
    return clamp(Math.floor(previousTokenCount), 0, Math.max(0, Math.floor(nextTokenCount)));
}

function createPromptChipOverlay({
    sourceEl,
    entry,
    index,
    promptTokenStrip,
    colorState,
    startRect
}) {
    const promptTokenEl = promptTokenStrip?.getTokenElement?.(index) || null;
    const promptTokenRect = promptTokenEl?.getBoundingClientRect?.() || null;

    const chip = document.createElement('span');
    chip.className = 'pass-intro-token-chip';
    chip.textContent = sourceEl?.textContent || promptTokenEl?.textContent || normalizeInlineTokenText(entry.displayLabel);
    applyTokenChipColors(chip, entry.chipEntry, index, { lookup: colorState.lookup });
    copyChipComputedStyle(sourceEl || promptTokenEl, chip);

    const targetRect = (promptTokenRect && promptTokenRect.width > 0 && promptTokenRect.height > 0)
        ? {
            left: promptTokenRect.left,
            top: promptTokenRect.top,
            right: promptTokenRect.right,
            bottom: promptTokenRect.bottom,
            width: promptTokenRect.width,
            height: promptTokenRect.height
        }
        : null;
    const fallbackStartRect = startRect || targetRect || {
        left: window.innerWidth * 0.5 - 24,
        top: window.innerHeight * 0.45 - 12,
        width: 48,
        height: 24,
        right: window.innerWidth * 0.5 + 24,
        bottom: window.innerHeight * 0.45 + 12
    };

    const targetWidth = Math.max(1, targetRect?.width || fallbackStartRect.width || 1);
    const targetHeight = Math.max(1, targetRect?.height || fallbackStartRect.height || 1);
    const startScaleX = clamp((fallbackStartRect.width || targetWidth) / targetWidth, 0.6, 2.8);
    const startScaleY = clamp((fallbackStartRect.height || targetHeight) / targetHeight, 0.75, 2.8);

    chip.style.left = `${fallbackStartRect.left + (fallbackStartRect.width || 0) / 2}px`;
    chip.style.top = `${fallbackStartRect.top + (fallbackStartRect.height || 0) / 2}px`;
    chip.style.width = `${targetWidth}px`;
    chip.style.height = `${targetHeight}px`;
    chip.style.opacity = '0';
    chip.style.transform = `translate(-50%, -50%) scale(${startScaleX.toFixed(4)}, ${startScaleY.toFixed(4)})`;
    chip.dataset.startScaleX = startScaleX.toFixed(4);
    chip.dataset.startScaleY = startScaleY.toFixed(4);
    chip.dataset.targetWidth = targetWidth.toFixed(2);

    return {
        chip,
        target: targetRect
            ? {
                x: targetRect.left + targetRect.width / 2,
                y: targetRect.top + targetRect.height / 2,
                scaleX: 1,
                scaleY: 1
            }
            : null
    };
}

export function initPassIntroOverlay({ activationSource, promptTokenStrip } = {}) {
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
    let currentTokenCount = 0;
    let activeHandoffRaf = null;
    let hideCleanupTimer = null;

    const clearHideCleanupTimer = () => {
        if (hideCleanupTimer) {
            clearTimeout(hideCleanupTimer);
            hideCleanupTimer = null;
        }
    };

    const clearHandoffInlineState = () => {
        dom.windowEl.style.opacity = '';
        dom.windowEl.style.transform = '';
        if (dom.scrimEl) dom.scrimEl.style.opacity = '';
    };

    const finalizeHiddenState = () => {
        clearHideCleanupTimer();
        dom.root.classList.remove('is-tokenized', 'is-handoff');
        dom.windowEl.classList.remove('is-transitioning');
        dom.textEl.classList.remove('is-tokenized', 'is-tokenizing', 'is-faded');
        clearHandoffInlineState();
        dom.tokenLayer.innerHTML = '';
        dom.textEl.textContent = '';
        if (dom.editorEl) dom.editorEl.scrollTop = 0;
        delete document.body.dataset.passIntroCommitted;
        document.body.classList.remove('pass-intro-active');
    };

    const hideOverlay = ({ immediate = false } = {}) => {
        if (activeHandoffRaf) {
            cancelAnimationFrame(activeHandoffRaf);
            activeHandoffRaf = null;
        }
        clearHideCleanupTimer();
        dom.root.dataset.visible = 'false';
        if (immediate) {
            finalizeHiddenState();
            return;
        }
        hideCleanupTimer = setTimeout(() => {
            finalizeHiddenState();
        }, OVERLAY_HIDE_CLEANUP_DELAY_MS);
    };

    const animateHandoffToPromptStrip = ({
        chips,
        targets
    }) => {
        if (!Array.isArray(chips) || !chips.length) return Promise.resolve();

        const trajectories = chips.map((chip, idx) => {
            const startX = Number.parseFloat(chip.style.left) || 0;
            const startY = Number.parseFloat(chip.style.top) || 0;
            const startScaleX = Number.parseFloat(chip.dataset.startScaleX) || 1;
            const startScaleY = Number.parseFloat(chip.dataset.startScaleY) || 1;
            const target = targets[idx] || {
                x: startX,
                y: startY,
                scaleX: 1,
                scaleY: 1
            };
            const dx = target.x - startX;
            const dy = target.y - startY;
            const distance = Math.hypot(dx, dy);
            const laneSkew = (idx / Math.max(1, chips.length - 1)) - 0.5;
            const baseArc = clamp(distance * 0.14, HANDOFF_MIN_ARC_PX, HANDOFF_MAX_ARC_PX);

            return {
                chip,
                startX,
                startY,
                startScaleX,
                startScaleY,
                targetX: target.x,
                targetY: target.y,
                delayMs: idx * HANDOFF_STAGGER_MS,
                durationMs: HANDOFF_BASE_DURATION_MS + clamp(distance * 0.12, 0, 160),
                arcPx: baseArc + laneSkew * 18,
                targetScaleX: Number.isFinite(target.scaleX) ? target.scaleX : 1,
                targetScaleY: Number.isFinite(target.scaleY) ? target.scaleY : 1,
                chipWidth: Number.parseFloat(chip.dataset.targetWidth) || 0
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
                dom.windowEl.style.opacity = String(lerp(1, 0.16, windowProgress));
                dom.windowEl.style.transform = `translateY(${lerp(0, -28, windowProgress).toFixed(2)}px) scale(${lerp(1, 0.96, windowProgress).toFixed(4)})`;
                if (dom.scrimEl) {
                    dom.scrimEl.style.opacity = String(lerp(1, 0.14, windowProgress));
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
                    const scaleX = lerp(entry.startScaleX, entry.targetScaleX, p);
                    const scaleY = lerp(entry.startScaleY, entry.targetScaleY, p);
                    const rotation = lerp(0, clamp((entry.targetX - entry.startX) / Math.max(96, entry.chipWidth), -8, 8), 1 - p);

                    entry.chip.style.left = `${x.toFixed(2)}px`;
                    entry.chip.style.top = `${y.toFixed(2)}px`;
                    entry.chip.style.transform = `translate(-50%, -50%) scale(${scaleX.toFixed(4)}, ${scaleY.toFixed(4)}) rotate(${rotation.toFixed(2)}deg)`;
                    entry.chip.style.opacity = '1';
                    entry.chip.style.filter = 'none';
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
        tokenLabels,
        presentation = 'typing',
        onHandoffCommit = null,
        onBeforeHide = null
    } = {}) => {
        if (disposed) return;

        const entries = buildTokenEntries({
            activationSource,
            laneCount,
            laneTokenIndices,
            tokenLabels
        });
        if (!entries.length) return;

        const promptEntries = buildPromptTokenChipEntries({
            tokenLabels: entries.map((entry) => entry.displayLabel),
            tokenIndices: entries.map((entry) => entry.tokenIndex),
            tokenIds: entries.map((entry) => entry.tokenId)
        });
        const colorState = resolvePromptTokenChipColorState(promptEntries);
        entries.forEach((entry, index) => {
            entry.chipEntry = promptEntries[index] || null;
        });

        const nextRawText = entries.map((entry) => entry.rawText).join('');
        const normalizedNextText = String(nextRawText ?? '').replace(/\r/g, '').replace(/\u00A0/g, ' ');
        const appendStartIndex = resolveAppendStartIndex(currentTokenCount, entries.length);
        const useTokenizedAppendPresentation = presentation === 'tokenized-append'
            && hasPlayedOnce
            && entries.length >= currentTokenCount
            && appendStartIndex <= entries.length;

        finalizeHiddenState();
        dom.root.dataset.visible = 'true';
        document.body.classList.add('pass-intro-active');
        hideLoadingOverlay();

        if (disposed) return;

        let inlineChipElements = [];
        if (useTokenizedAppendPresentation) {
            currentRawText = normalizedNextText;
            dom.root.classList.add('is-tokenized');
            dom.textEl.classList.add('is-tokenized');
            inlineChipElements = renderInlinePromptTokens(
                dom.textEl,
                entries,
                colorState,
                promptTokenStrip,
                {
                    delayResolver: (index) => (
                        index < appendStartIndex ? 0 : (index - appendStartIndex) * APPEND_TOKEN_STAGGER_MS
                    )
                }
            );
            inlineChipElements.forEach((chip, index) => {
                if (index < appendStartIndex) {
                    chip.classList.add('is-visible');
                    return;
                }
                chip.classList.add('is-appended');
            });
            syncEditorScroll(dom.editorEl);

            await nextFrame();
            if (disposed) return;

            for (let i = appendStartIndex; i < inlineChipElements.length; i += 1) {
                inlineChipElements[i].classList.add('is-visible');
            }

            const appendedCount = Math.max(0, inlineChipElements.length - appendStartIndex);
            const appendTailMs = appendedCount > 0
                ? APPEND_TOKEN_SETTLE_MS + APPEND_TOKEN_STAGGER_MS * Math.max(0, appendedCount - 1)
                : 120;
            await delay(appendTailMs);
            if (disposed) return;

            await delay(APPEND_TOKEN_HOLD_MS);
            if (disposed) return;
        } else {
            const prefixLen = commonPrefixLength(currentRawText, normalizedNextText);
            let typedText = currentRawText;

            if (prefixLen < typedText.length) {
                while (typedText.length > prefixLen) {
                    if (disposed) return;
                    typedText = typedText.slice(0, -1);
                    dom.textEl.textContent = typedText;
                    syncEditorScroll(dom.editorEl);
                    await delay(TYPE_DELETE_MS);
                }
            } else {
                dom.textEl.textContent = typedText;
                syncEditorScroll(dom.editorEl);
            }

            const suffix = normalizedNextText.slice(prefixLen);
            const typeBaseMs = hasPlayedOnce ? NEXT_PASS_TYPE_BASE_MS : FIRST_PASS_TYPE_BASE_MS;
            for (let i = 0; i < suffix.length; i += 1) {
                if (disposed) return;
                typedText += suffix[i];
                dom.textEl.textContent = typedText;
                syncEditorScroll(dom.editorEl);
                await delay(resolveTypingDelayMs(suffix[i], typeBaseMs));
            }
            if (!suffix.length) {
                await delay(80);
            }

            currentRawText = normalizedNextText;
            dom.root.classList.add('is-tokenized');
            await delay(TYPE_SETTLE_MS);
            if (disposed) return;

            dom.textEl.classList.add('is-tokenized');
            inlineChipElements = renderInlinePromptTokens(dom.textEl, entries, colorState, promptTokenStrip);
            syncEditorScroll(dom.editorEl);

            await nextFrame();
            if (disposed) return;

            inlineChipElements.forEach((chip) => {
                chip.classList.add('is-visible');
            });

            const tokenizeTailMs = TOKENIZE_IN_PLACE_DURATION_MS
                + TOKENIZE_STAGGER_MS * Math.max(0, inlineChipElements.length - 1);
            await delay(tokenizeTailMs);
            if (disposed) return;

            await delay(TOKENIZE_HOLD_MS);
            if (disposed) return;
        }

        const overlayEntries = entries.map((entry, idx) => {
            const sourceEl = inlineChipElements[idx] || null;
            const sourceRect = sourceEl?.getBoundingClientRect?.() || null;
            const overlay = createPromptChipOverlay({
                sourceEl,
                entry,
                index: idx,
                promptTokenStrip,
                colorState,
                startRect: (sourceRect && sourceRect.width > 0 && sourceRect.height > 0)
                    ? {
                        left: sourceRect.left,
                        top: sourceRect.top,
                        right: sourceRect.right,
                        bottom: sourceRect.bottom,
                        width: sourceRect.width,
                        height: sourceRect.height
                    }
                    : null
            });
            overlay.chip.style.transition = 'none';
            overlay.chip.style.opacity = '1';
            dom.tokenLayer.appendChild(overlay.chip);
            return overlay;
        });

        await nextFrame();
        if (disposed) return;

        dom.textEl.classList.add('is-faded');
        dom.root.classList.add('is-handoff');
        dom.windowEl.classList.add('is-transitioning');

        const chips = overlayEntries.map((entry) => entry.chip);
        const targets = overlayEntries.map((entry) => (
            entry.target || {
                x: Number.parseFloat(entry.chip.style.left) || (window.innerWidth * 0.5),
                y: Number.parseFloat(entry.chip.style.top) || (window.innerHeight * 0.78),
                scaleX: 1,
                scaleY: 1
            }
        ));

        await animateHandoffToPromptStrip({
            chips,
            targets
        });
        dom.tokenLayer.innerHTML = '';
        document.body.dataset.passIntroCommitted = 'true';
        if (typeof onHandoffCommit === 'function') {
            onHandoffCommit();
        }
        await delay(40);
        if (disposed) return;
        if (typeof onBeforeHide === 'function') {
            await onBeforeHide();
        }
        if (disposed) return;

        hasPlayedOnce = true;
        currentTokenCount = entries.length;
        hideOverlay();
    };

    const dispose = () => {
        if (disposed) return;
        disposed = true;
        hideOverlay({ immediate: true });
    };

    return { play, dispose };
}

export default initPassIntroOverlay;
