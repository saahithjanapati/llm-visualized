const STATS_JS_SRC = 'https://cdn.jsdelivr.net/npm/stats.js@0.17.0/build/stats.min.js';
const STATS_JS_SCRIPT_SELECTOR = 'script[data-stats-js-loader="true"]';

let statsJsPromise = null;

function getStatsConstructor() {
    if (typeof window === 'undefined') return null;
    return typeof window.Stats === 'function' ? window.Stats : null;
}

export function loadStatsConstructor() {
    const existingConstructor = getStatsConstructor();
    if (existingConstructor) return Promise.resolve(existingConstructor);
    if (typeof document === 'undefined') return Promise.resolve(null);
    if (statsJsPromise) return statsJsPromise;

    statsJsPromise = new Promise((resolve, reject) => {
        let script = document.querySelector(STATS_JS_SCRIPT_SELECTOR);
        const cleanup = () => {
            script?.removeEventListener('load', handleLoad);
            script?.removeEventListener('error', handleError);
        };
        const settleWithConstructor = () => {
            const StatsConstructor = getStatsConstructor();
            if (StatsConstructor) {
                resolve(StatsConstructor);
                return true;
            }
            return false;
        };
        const handleLoad = () => {
            cleanup();
            if (settleWithConstructor()) return;
            statsJsPromise = null;
            reject(new Error('Stats.js loaded without exposing window.Stats.'));
        };
        const handleError = () => {
            cleanup();
            statsJsPromise = null;
            reject(new Error('Failed to load Stats.js.'));
        };

        if (script?.dataset.loaded === 'true' && settleWithConstructor()) {
            cleanup();
            return;
        }

        if (!script) {
            script = document.createElement('script');
            script.src = STATS_JS_SRC;
            script.async = true;
            script.dataset.statsJsLoader = 'true';
            document.head.appendChild(script);
        }

        script.addEventListener('load', () => {
            script.dataset.loaded = 'true';
            handleLoad();
        }, { once: true });
        script.addEventListener('error', handleError, { once: true });
    }).finally(() => {
        if (getStatsConstructor()) return;
        statsJsPromise = null;
    });

    return statsJsPromise;
}
