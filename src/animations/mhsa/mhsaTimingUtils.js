const SKIP_DELAY_SCALE = 0.03;
const SKIP_DURATION_SCALE = 0.05;
const SKIP_DELAY_MIN_MS = 1;
const SKIP_DELAY_MAX_MS = 5;
const SKIP_DURATION_MIN_MS = 2;
const SKIP_DURATION_MAX_MS = 14;

export function resolveSkipDelay(skipToEndActive, delayMs) {
    const clamped = Math.max(0, Number(delayMs) || 0);
    if (!skipToEndActive) return clamped;
    if (clamped <= 0) return 0;
    const scaled = clamped * SKIP_DELAY_SCALE;
    return Math.min(SKIP_DELAY_MAX_MS, Math.max(SKIP_DELAY_MIN_MS, scaled));
}

export function resolveSkipDuration(skipToEndActive, durationMs) {
    const clamped = Math.max(0, Number(durationMs) || 0);
    if (!skipToEndActive) return clamped;
    if (clamped <= 0) return 0;
    const scaled = clamped * SKIP_DURATION_SCALE;
    return Math.min(SKIP_DURATION_MAX_MS, Math.max(SKIP_DURATION_MIN_MS, scaled));
}

export function scheduleAfterDelay({
    callback,
    delayMs,
    skipToEndActive = false,
    scheduledDelayTweens,
    scheduledTimeoutIds,
    tweenLib,
    onError
} = {}) {
    if (typeof callback !== 'function') return () => {};
    const clampedDelay = resolveSkipDelay(skipToEndActive, delayMs);
    const logError = typeof onError === 'function' ? onError : null;

    if (tweenLib && typeof tweenLib.Tween === 'function') {
        const state = { t: 0 };
        const tween = new tweenLib.Tween(state)
            .to({ t: 1 }, clampedDelay)
            .onComplete(() => {
                if (scheduledDelayTweens) {
                    scheduledDelayTweens.delete(tween);
                }
                try {
                    callback();
                } catch (err) {
                    if (logError) logError(err);
                }
            })
            .onStop(() => {
                if (scheduledDelayTweens) {
                    scheduledDelayTweens.delete(tween);
                }
            })
            .start();
        if (scheduledDelayTweens) {
            scheduledDelayTweens.add(tween);
        }
        return () => {
            try {
                tween.stop();
            } catch (_) { /* ignore */ }
        };
    }

    const timeoutId = setTimeout(() => {
        if (scheduledTimeoutIds) {
            scheduledTimeoutIds.delete(timeoutId);
        }
        callback();
    }, clampedDelay);
    if (scheduledTimeoutIds) {
        scheduledTimeoutIds.add(timeoutId);
    }
    return () => {
        clearTimeout(timeoutId);
        if (scheduledTimeoutIds) {
            scheduledTimeoutIds.delete(timeoutId);
        }
    };
}

export function clearScheduledDelays(scheduledDelayTweens, scheduledTimeoutIds) {
    if (scheduledDelayTweens) {
        scheduledDelayTweens.forEach((tween) => {
            try { tween.stop(); } catch (_) { /* ignore */ }
        });
        scheduledDelayTweens.clear();
    }
    if (scheduledTimeoutIds) {
        scheduledTimeoutIds.forEach((id) => clearTimeout(id));
        scheduledTimeoutIds.clear();
    }
}
