const DEFAULT_SAMPLE_INTERVAL_MS = 500;

function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
}

export const perfStats = {
    enabled: false,
    sampleIntervalMs: DEFAULT_SAMPLE_INTERVAL_MS,
    _frameStart: 0,
    _lastSampleTime: 0,
    _frameCount: 0,
    _accum: {
        update: 0,
        tween: 0,
        render: 0,
        total: 0,
        trailUpdates: 0,
        colorUpdates: 0,
        colorCacheHits: 0,
        raycasts: 0,
        selectRaycasts: 0,
        raycastIntersects: 0,
        vectorUpdates: 0,
    },
    _gauges: {},
    _snapshot: null,
    _snapshotVersion: 0,

    beginFrame(now) {
        if (!this.enabled) return;
        if (!isFiniteNumber(now)) return;
        this._frameStart = now;
    },

    addTime(key, ms) {
        if (!this.enabled) return;
        if (!isFiniteNumber(ms) || !key) return;
        if (!Object.prototype.hasOwnProperty.call(this._accum, key)) {
            this._accum[key] = 0;
        }
        this._accum[key] += ms;
    },

    inc(key, amount = 1) {
        if (!this.enabled) return;
        if (!key) return;
        const delta = isFiniteNumber(amount) ? amount : 1;
        if (!Object.prototype.hasOwnProperty.call(this._accum, key)) {
            this._accum[key] = 0;
        }
        this._accum[key] += delta;
    },

    setGauge(key, value) {
        if (!this.enabled) return;
        if (!key) return;
        this._gauges[key] = value;
    },

    endFrame(now) {
        if (!this.enabled) return;
        if (!isFiniteNumber(now)) return;

        const totalMs = now - (this._frameStart || now);
        if (isFiniteNumber(totalMs)) {
            this._accum.total += totalMs;
        }
        this._frameCount += 1;

        if (!this._lastSampleTime) {
            this._lastSampleTime = now;
            return;
        }

        const elapsed = now - this._lastSampleTime;
        if (elapsed < this.sampleIntervalMs) return;

        const frames = Math.max(1, this._frameCount);
        const perFrame = (value) => value / frames;
        const colorUpdates = this._accum.colorUpdates || 0;
        const colorCacheHits = this._accum.colorCacheHits || 0;

        this._snapshot = {
            fps: elapsed > 0 ? (frames * 1000) / elapsed : 0,
            totalMs: perFrame(this._accum.total || 0),
            updateMs: perFrame(this._accum.update || 0),
            tweenMs: perFrame(this._accum.tween || 0),
            renderMs: perFrame(this._accum.render || 0),
            trailUpdates: perFrame(this._accum.trailUpdates || 0),
            colorUpdates: perFrame(colorUpdates),
            colorCacheHitRate: colorUpdates > 0 ? (colorCacheHits / colorUpdates) * 100 : 0,
            raycasts: perFrame(this._accum.raycasts || 0),
            selectRaycasts: perFrame(this._accum.selectRaycasts || 0),
            raycastIntersects: perFrame(this._accum.raycastIntersects || 0),
            vectorUpdates: perFrame(this._accum.vectorUpdates || 0),
            gauges: { ...this._gauges },
        };
        this._snapshotVersion += 1;

        Object.keys(this._accum).forEach((key) => {
            this._accum[key] = 0;
        });
        this._frameCount = 0;
        this._lastSampleTime = now;
    },

    getSnapshot() {
        return this._snapshot;
    },

    getSnapshotVersion() {
        return this._snapshotVersion;
    },
};
