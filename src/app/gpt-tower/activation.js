import { CaptureActivationSource } from '../../data/CaptureActivationSource.js';
import { precomputeActivationCaches } from '../../utils/activationPrecompute.js';
import { DEFAULT_CAPTURE_FILE, resolveCaptureAssetUrl } from './runtimeAssetUrls.js';

function getFullTokenMode() {
    if (typeof window === 'undefined') return false;
    const path = window.location.pathname.replace(/\/+$/, '');
    return path.endsWith('/full');
}

export async function loadActivationState({
    defaultLaneCount,
    defaultPromptLanes,
    promptTokens,
    positionTokens,
    layerCount,
    statusElement
}) {
    let activationSource = null;
    let laneTokenIndices = null;
    let laneCount = defaultLaneCount;
    const isFullTokenMode = getFullTokenMode();

    const setLoadingStatus = (text) => {
        if (statusElement) statusElement.textContent = text;
    };

    try {
        const params = new URLSearchParams(window.location.search);
        const captureFile = params.get('capture') || params.get('file') || DEFAULT_CAPTURE_FILE;
        const captureUrl = resolveCaptureAssetUrl(captureFile);
        activationSource = await CaptureActivationSource.load(captureUrl);
        const tokensInCapture = typeof activationSource.getTokenCount === 'function'
            ? activationSource.getTokenCount()
            : 0;
        const safeTokenCount = Math.max(0, tokensInCapture || 0);
        const desiredLanes = Math.max(1, safeTokenCount || laneCount);
        if (isFullTokenMode) {
            laneCount = desiredLanes;
        } else {
            const cappedLanes = safeTokenCount
                ? Math.min(safeTokenCount, defaultPromptLanes)
                : defaultPromptLanes;
            laneCount = Math.min(desiredLanes, cappedLanes);
        }
        laneTokenIndices = activationSource.getLaneTokenIndices(laneCount);
    } catch (err) {
        console.warn('Capture data unavailable; falling back to random vectors.', err);
    }

    if (!activationSource) {
        laneCount = defaultPromptLanes;
    }

    if (activationSource && laneTokenIndices) {
        try {
            setLoadingStatus('Preparing activation cache...');
            await precomputeActivationCaches(activationSource, {
                layerCount,
                laneTokenIndices,
                onProgress: ({ message }) => {
                    if (message) {
                        setLoadingStatus(message);
                    }
                }
            });
        } catch (err) {
            console.warn('Activation cache precompute failed; continuing without warm cache.', err);
        }
    }

    const tokenLabels = isFullTokenMode && activationSource && laneTokenIndices
        ? laneTokenIndices.map((idx) => activationSource.getTokenString(idx) || '')
        : promptTokens;
    const positionLabels = isFullTokenMode && activationSource && laneTokenIndices
        ? laneTokenIndices.map((idx) => String(idx + 1))
        : positionTokens;

    return {
        activationSource,
        laneTokenIndices,
        laneCount,
        tokenLabels,
        positionLabels,
        isFullTokenMode
    };
}
