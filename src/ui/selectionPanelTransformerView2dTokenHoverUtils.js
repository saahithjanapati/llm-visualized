import { resolveHoverTokenChipSyncEntries } from '../engine/coreHoverTokenContext.js';
import {
    TOKEN_CHIP_HOVER_SYNC_EVENT,
    dispatchTokenChipHoverSync,
    matchesFocusVisibleTarget,
    normalizeTokenChipEntry,
    normalizeTokenChipEntries,
    tokenChipEntryListsMatch,
    tokenChipEntriesMatch
} from './tokenChipHoverSync.js';

export const TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE = 'transformer-view2d';

const TRANSFORMER_VIEW2D_TOKEN_CHIP_SELECTOR = '.detail-transformer-view2d-token-strip__token';
const TRANSFORMER_VIEW2D_TOKEN_CHIP_TARGET_SELECTOR = `${TRANSFORMER_VIEW2D_TOKEN_CHIP_SELECTOR}[data-token-nav="true"]`;
const TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS = 'canvas';
const TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP = 'strip';

function tokenEntriesEquivalent(a, b) {
    return tokenChipEntryListsMatch(a, b);
}

export function extractTransformerView2dTokenChipEntry(chip) {
    if (!chip) return null;
    return normalizeTokenChipEntry({
        tokenLabel: chip.dataset?.tokenText || chip.textContent,
        tokenIndex: chip.dataset?.tokenIndex,
        tokenId: chip.dataset?.tokenId
    });
}

function resolveTransformerView2dTokenEntryFromHoverInfo(info = null) {
    return normalizeTokenChipEntry({
        tokenIndex: info?.tokenIndex ?? info?.queryTokenIndex ?? info?.keyTokenIndex,
        tokenId: info?.tokenId ?? info?.token_id,
        tokenLabel: info?.tokenLabel
            || info?.queryTokenLabel
            || info?.keyTokenLabel
            || info?.tokenText
            || info?.token
    });
}

function resolveTransformerView2dTokenEntriesFromHoverInfo(info = null) {
    const entries = normalizeTokenChipEntries([
        {
            tokenIndex: info?.tokenIndex ?? info?.queryTokenIndex,
            tokenId: info?.tokenId ?? info?.queryTokenId ?? info?.token_id,
            tokenLabel: info?.tokenLabel
                || info?.queryTokenLabel
                || info?.tokenText
                || info?.token
        },
        {
            tokenIndex: info?.keyTokenIndex,
            tokenId: info?.keyTokenId,
            tokenLabel: info?.keyTokenLabel
        }
    ]);
    if (entries.length) return entries;
    const fallbackEntry = resolveTransformerView2dTokenEntryFromHoverInfo(info);
    return fallbackEntry ? [fallbackEntry] : [];
}

function resolveTransformerView2dTokenEntriesFromSharedHoverPayload(payload = null) {
    const label = typeof payload?.label === 'string' && payload.label.length
        ? payload.label
        : (
            payload?.info?.activationData?.label
            || payload?.activationData?.label
            || payload?.info?.label
            || ''
        );
    const stage = payload?.info?.activationData?.stage
        || payload?.activationData?.stage
        || payload?.info?.stage
        || payload?.stage
        || '';
    const hasSharedContext = Boolean(
        (typeof label === 'string' && label.length)
        || (typeof stage === 'string' && stage.length)
        || payload?.object
        || payload?.hit?.object
    );
    if (!hasSharedContext) {
        return null;
    }
    return resolveHoverTokenChipSyncEntries({
        label,
        info: payload?.info || payload || null,
        object: payload?.object || payload?.hit?.object || null
    });
}

export function resolveTransformerView2dTokenEntriesFromHoverPayload(payload = null) {
    const sharedEntries = resolveTransformerView2dTokenEntriesFromSharedHoverPayload(payload);
    if (sharedEntries) return sharedEntries;
    const sources = [
        payload?.tokenEntries,
        payload?.tokenEntry,
        payload?.info?.activationData,
        payload?.activationData,
        payload?.info,
        payload
    ];
    for (const source of sources) {
        const entries = normalizeTokenChipEntries(source);
        if (entries.length) return entries;
        const infoEntries = resolveTransformerView2dTokenEntriesFromHoverInfo(source);
        if (infoEntries.length) return infoEntries;
    }
    return [];
}

export function resolveTransformerView2dTokenEntryFromHoverPayload(payload = null) {
    return resolveTransformerView2dTokenEntriesFromHoverPayload(payload)[0] || null;
}

export function resolveTransformerView2dTokenEntriesFromResidualHoverPayload(payload = null) {
    const sharedEntries = resolveTransformerView2dTokenEntriesFromSharedHoverPayload(payload);
    if (sharedEntries) return sharedEntries;
    const info = payload?.info || payload || null;
    return resolveTransformerView2dTokenEntriesFromHoverInfo(info);
}

export function resolveTransformerView2dTokenEntryFromResidualHoverPayload(payload = null) {
    return resolveTransformerView2dTokenEntriesFromResidualHoverPayload(payload)[0]
        || resolveTransformerView2dTokenEntryFromHoverInfo(payload?.info || payload || null);
}

export function createTransformerView2dTokenHoverSync({ container = null } = {}) {
    let localEntries = [];
    let localSource = '';
    let mirroredEntries = [];

    const applyActiveState = () => {
        if (!container || typeof container.querySelectorAll !== 'function') return;
        let hasFocusedChip = false;
        container.querySelectorAll(TRANSFORMER_VIEW2D_TOKEN_CHIP_SELECTOR).forEach((chip) => {
            const canNavigate = chip.dataset.tokenNav === 'true';
            const chipEntry = canNavigate ? extractTransformerView2dTokenChipEntry(chip) : null;
            const isLocallyActive = !!chipEntry && localEntries.some((candidate) => (
                tokenChipEntriesMatch(chipEntry, candidate)
            ));
            const isMirroredActive = !!chipEntry && mirroredEntries.some((candidate) => (
                tokenChipEntriesMatch(chipEntry, candidate)
            ));
            const isActive = isLocallyActive || isMirroredActive;
            if (isActive) hasFocusedChip = true;
            chip.classList.toggle('is-token-chip-active', isActive);
            chip.classList.toggle('is-token-chip-hover-synced', isMirroredActive);
            chip.dataset.tokenActive = isActive ? 'true' : 'false';
        });
        container.dataset.tokenFocusActive = hasFocusedChip ? 'true' : 'false';
    };

    const emitLocalEntries = () => {
        dispatchTokenChipHoverSync(localEntries, {
            active: localEntries.length > 0,
            source: TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE
        });
    };

    const setLocalEntries = (entries, source = '', { emit = true } = {}) => {
        const normalizedEntries = normalizeTokenChipEntries(entries);
        const nextSource = normalizedEntries.length ? String(source || '').trim() : '';
        const didChange = localSource !== nextSource || !tokenEntriesEquivalent(localEntries, normalizedEntries);
        localEntries = normalizedEntries;
        localSource = nextSource;
        if (didChange && emit) {
            emitLocalEntries();
        }
        applyActiveState();
    };

    const clearLocalEntry = (source = '', { emit = true } = {}) => {
        const safeSource = String(source || '').trim();
        if (safeSource.length && localSource !== safeSource) {
            applyActiveState();
            return;
        }
        if (!localEntries.length && !localSource) {
            applyActiveState();
            return;
        }
        localEntries = [];
        localSource = '';
        if (emit) {
            emitLocalEntries();
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
        setLocalEntries(
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
            setLocalEntries(
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
        if (!matchesFocusVisibleTarget(chip)) return;
        setLocalEntries(
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
            setLocalEntries(
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
        mirroredEntries = detail.active ? normalizeTokenChipEntries(detail) : [];
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
        setCanvasEntry: (entry, options = {}) => setLocalEntries(
            entry,
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS,
            options
        ),
        setCanvasEntryFromResidualHoverPayload: (payload, options = {}) => setLocalEntries(
            resolveTransformerView2dTokenEntriesFromResidualHoverPayload(payload),
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS,
            options
        ),
        setCanvasEntryFromHoverPayload: (payload, options = {}) => setLocalEntries(
            resolveTransformerView2dTokenEntriesFromHoverPayload(payload),
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS,
            options
        ),
        clearCanvasEntry: (options = {}) => clearLocalEntry(
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_CANVAS,
            options
        ),
        setStripEntry: (entry, options = {}) => setLocalEntries(
            entry,
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP,
            options
        ),
        clearStripEntry: (options = {}) => clearLocalEntry(
            TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE_STRIP,
            options
        ),
        clear: ({ emit = true, resetMirrored = true } = {}) => {
            clearLocalEntry('', { emit });
            if (resetMirrored && mirroredEntries.length) {
                mirroredEntries = [];
                applyActiveState();
            }
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
            mirroredEntries = [];
            applyActiveState();
        }
    };
}
