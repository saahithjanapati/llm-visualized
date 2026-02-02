import { perfStats } from '../utils/perfStats.js';

function formatNumber(value, decimals = 1) {
    if (!Number.isFinite(value)) return '--';
    return value.toFixed(decimals);
}

function buildOverlayText(snapshot) {
    const gauges = snapshot.gauges || {};
    const tweens = Number.isFinite(gauges.tweens) ? Math.round(gauges.tweens) : '--';
    const lines = [
        'Perf',
        `FPS: ${formatNumber(snapshot.fps, 1)} | Frame: ${formatNumber(snapshot.totalMs, 1)} ms`,
        `Update: ${formatNumber(snapshot.updateMs, 1)} | Tween: ${formatNumber(snapshot.tweenMs, 1)} | Render: ${formatNumber(snapshot.renderMs, 1)}`,
        `Trails: ${formatNumber(snapshot.trailUpdates, 1)} /f | Colors: ${formatNumber(snapshot.colorUpdates, 1)} /f | Cache: ${formatNumber(snapshot.colorCacheHitRate, 0)}%`,
        `Raycast: ${formatNumber(snapshot.raycasts, 2)} /f | Select: ${formatNumber(snapshot.selectRaycasts, 2)} /f | Hits: ${formatNumber(snapshot.raycastIntersects, 1)} /f`,
        `Tweens: ${tweens}`,
    ];
    return lines.join('\n');
}

export function initPerfOverlay({ intervalMs = 250 } = {}) {
    if (typeof document === 'undefined') {
        return { dispose: () => {} };
    }

    const overlay = document.createElement('div');
    overlay.id = 'perfOverlay';
    Object.assign(overlay.style, {
        position: 'fixed',
        left: '0px',
        top: '56px',
        padding: '8px 10px',
        fontFamily: 'monospace',
        fontSize: '12px',
        color: '#fff',
        background: 'rgba(20,20,20,0.35)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        borderRadius: '8px',
        pointerEvents: 'none',
        zIndex: 6,
        whiteSpace: 'pre',
    });
    overlay.textContent = 'Perf\n...';
    document.body.appendChild(overlay);

    perfStats.enabled = true;

    let lastVersion = -1;
    const timer = setInterval(() => {
        const version = perfStats.getSnapshotVersion();
        if (version === lastVersion) return;
        const snapshot = perfStats.getSnapshot();
        if (!snapshot) return;
        overlay.textContent = buildOverlayText(snapshot);
        lastVersion = version;
    }, Math.max(100, intervalMs));

    return {
        dispose: () => {
            clearInterval(timer);
            if (overlay.parentElement) overlay.parentElement.removeChild(overlay);
            perfStats.enabled = false;
        }
    };
}
