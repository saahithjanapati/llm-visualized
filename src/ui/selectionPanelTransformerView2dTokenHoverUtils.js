import {
    TOKEN_CHIP_HOVER_SYNC_EVENT,
    dispatchTokenChipHoverSync,
    normalizeTokenChipEntry,
    tokenChipEntriesMatch
} from './tokenChipHoverSync.js';

export const TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE = 'transformer-view2d';

const TRANSFORMER_VIEW2D_TOKEN_CHIP_SELECTOR = '.detail-transformer-view2d-token-strip__token';
const TRANSFORMER_VIEW2D_TOKEN_CHIP_TARGET_SELECTOR = `${TRANSFORMER_VIEW2D_TOKEN_CHIP_SELECTOR}[data-token-nav="true"]`;
const TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS = 'canvas';
const TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP = 'strip';

function tokenEntriesEquivalent(a, b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return tokenChipEntriesMatch(a, b);
}

export function extractTransformerView2dTokenChipEntry(chip) {
    if (!chip) return null;
    return normalizeTokenChipEntry({
        tokenLabel: chip.dataset?.tokenText || chip.textContent,
        tokenIndex: chip.dataset?.tokenIndex,
        tokenId: chip.dataset?.tokenId
    });
}

export function resolveTransformerView2dTokenEntryFromResidualHoverPayload(payload = null) {
    const info = payload?.info || payload || null;
    return normalizeTokenChipEntry({
        tokenIndex: info?.tokenIndex,
        tokenId: info?.tokenId,
        tokenLabel: info?.tokenLabel
    });
}

export function createTransformerView2dTokenHoverSync({ container = null } = {}) {
    let localEntry = null;
    let localSource = '';
    let mirroredEntry = null;

    const applyActiveState = () => {
        if (!container || typeof container.querySelectorAll !== 'function') return;
        container.querySelectorAll(TRANSFORMER_VIEW2D_TOKEN_CHIP_SELECTOR).forEach((chip) => {
            const canNavigate = chip.dataset.tokenNav === 'true';
            const chipEntry = canNavigate ? extractTransformerView2dTokenChipEntry(chip) : null;
            const isActive = !!chipEntry && (
                tokenChipEntriesMatch(chipEntry, localEntry)
                || tokenChipEntriesMatch(chipEntry, mirroredEntry)
            );
            chip.classList.toggle('is-token-chip-active', isActive);
            chip.dataset.tokenActive = isActive ? 'true' : 'false';
        });
    };

    const emitLocalEntry = () => {
        dispatchTokenChipHoverSync(localEntry, {
            active: !!localEntry,
            source: TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE
        });
    };

    const setLocalEntry = (entry, source = '', { emit = true } = {}) => {
        const normalizedEntry = normalizeTokenChipEntry(entry);
        const nextSource = normalizedEntry ? String(source || '').trim() : '';
        const didChange = localSource !== nextSource || !tokenEntriesEquivalent(localEntry, normalizedEntry);
        localEntry = normalizedEntry;
        localSource = nextSource;
        if (didChange && emit) {
            emitLocalEntry();
        }
        applyActiveState();
    };

    const clearLocalEntry = (source = '', { emit = true } = {}) => {
        const safeSource = String(source || '').trim();
        if (safeSource.length && localSource !== safeSource) {
            applyActiveState();
            return;
        }
        if (!localEntry && !localSource) {
            applyActiveState();
            return;
        }
        localEntry = null;
        localSource = '';
        if (emit) {
            emitLocalEntry();
        }
        applyActiveState();
    };

    const resolveChipTarget = (target) => {
        if (!container || !target || typeof target.closest !== 'function') return null;
        const chip = target.closest(TRANSFORMER_VIEW2D_TOKEN_CHIP_TARGET_SELECTOR);
        if (!chip || !container.contains(chip)) return null;
        return chip;
    };

    const handlePointerOver = (event) => {
        const chip = resolveChipTarget(event?.target);
        if (!chip) return;
        setLocalEntry(
            extractTransformerView2dTokenChipEntry(chip),
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP,
            { emit: true }
        );
    };

    const handlePointerOut = (event) => {
        const fromChip = resolveChipTarget(event?.target);
        if (!fromChip) return;
        const toChip = resolveChipTarget(event?.relatedTarget);
        if (toChip) {
            setLocalEntry(
                extractTransformerView2dTokenChipEntry(toChip),
                TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP,
                { emit: true }
            );
            return;
        }
        clearLocalEntry(TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP, { emit: true });
    };

    const handleFocusIn = (event) => {
        const chip = resolveChipTarget(event?.target);
        if (!chip) return;
        setLocalEntry(
            extractTransformerView2dTokenChipEntry(chip),
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP,
            { emit: true }
        );
    };

    const handleFocusOut = (event) => {
        const fromChip = resolveChipTarget(event?.target);
        if (!fromChip) return;
        const toChip = resolveChipTarget(event?.relatedTarget);
        if (toChip) {
            setLocalEntry(
                extractTransformerView2dTokenChipEntry(toChip),
                TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP,
                { emit: true }
            );
            return;
        }
        clearLocalEntry(TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP, { emit: true });
    };

    const handleTokenChipHoverSync = (event) => {
        const detail = event?.detail || null;
        if (!detail || detail.source === TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE) return;
        mirroredEntry = detail.active ? normalizeTokenChipEntry(detail) : null;
        applyActiveState();
    };

    container?.addEventListener?.('pointerover', handlePointerOver);
    container?.addEventListener?.('pointerout', handlePointerOut);
    container?.addEventListener?.('focusin', handleFocusIn);
    container?.addEventListener?.('focusout', handleFocusOut);
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, handleTokenChipHoverSync);
    }

    return {
        applyState: applyActiveState,
        setCanvasEntry: (entry, options = {}) => setLocalEntry(
            entry,
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS,
            options
        ),
        setCanvasEntryFromResidualHoverPayload: (payload, options = {}) => setLocalEntry(
            resolveTransformerView2dTokenEntryFromResidualHoverPayload(payload),
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS,
            options
        ),
        clearCanvasEntry: (options = {}) => clearLocalEntry(
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS,
            options
        ),
        setStripEntry: (entry, options = {}) => setLocalEntry(
            entry,
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP,
            options
        ),
        clearStripEntry: (options = {}) => clearLocalEntry(
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP,
            options
        ),
        clear: ({ emit = true } = {}) => {
            clearLocalEntry('', { emit });
        },
        dispose: ({ emit = true } = {}) => {
            if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
                window.removeEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, handleTokenChipHoverSync);
            }
            container?.removeEventListener?.('pointerover', handlePointerOver);
            container?.removeEventListener?.('pointerout', handlePointerOut);
            container?.removeEventListener?.('focusin', handleFocusIn);
            container?.removeEventListener?.('focusout', handleFocusOut);
            clearLocalEntry('', { emit });
            mirroredEntry = null;
            applyActiveState();
        }
    };
}
