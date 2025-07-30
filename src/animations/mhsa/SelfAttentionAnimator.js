export class SelfAttentionAnimator {
    /**
     * @param {MHSAAnimation} ctx – parent animation instance.
     * @param {number} durationMs – how long the placeholder pause should last.
     */
    constructor(ctx, durationMs = 3000) {
        this.ctx = ctx;
        this.duration = durationMs;
        this.phase = 'waiting'; // 'waiting' | 'running' | 'complete'
        this._callbacks = [];
    }

    /**
     * Start the self-attention phase. Accepts an optional callback invoked once
     * the phase completes.  Calling `start` multiple times before completion
     * just queues additional callbacks.
     */
    start(onDone) {
        if (typeof onDone === 'function') this._callbacks.push(onDone);

        if (this.phase === 'complete') {
            // Phase already finished – invoke immediately.
            this._flushCallbacks();
            return;
        }
        if (this.phase === 'running') {
            // Already running; callback queued above – nothing else to do.
            return;
        }

        this.phase = 'running';
        console.log('SelfAttentionAnimator: placeholder phase started');

        setTimeout(() => {
            this.phase = 'complete';
            console.log('SelfAttentionAnimator: placeholder phase complete');
            this._flushCallbacks();
        }, this.duration);
    }

    _flushCallbacks() {
        const list = this._callbacks.splice(0, this._callbacks.length);
        list.forEach(cb => {
            try { cb && cb(); } catch(err) { console.error(err); }
        });
    }
} 