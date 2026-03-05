import { afterEach, describe, expect, it, vi } from 'vitest';
import { logLayerNormVectorDump } from '../src/engine/layers/gpt2LayerDebugUtils.js';

describe('gpt2LayerDebugUtils', () => {
    afterEach(() => {
        delete globalThis.__LN_VECTOR_DEBUG;
        vi.restoreAllMocks();
    });

    it('logs LN vector dumps only once per lane+kind', () => {
        const lane = {
            laneIndex: 1,
            tokenIndex: 2,
            tokenLabel: 'hello'
        };
        const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
        const groupSpy = vi.spyOn(console, 'groupCollapsed').mockImplementation(() => {});
        const endSpy = vi.spyOn(console, 'groupEnd').mockImplementation(() => {});

        logLayerNormVectorDump({
            layerIndex: 0,
            kind: 'ln1',
            lane,
            vectors: { normalizedSaved: [1, 2, 3] }
        });
        logLayerNormVectorDump({
            layerIndex: 0,
            kind: 'ln1',
            lane,
            vectors: { normalizedSaved: [1, 2, 3] }
        });

        expect(groupSpy).toHaveBeenCalledTimes(1);
        expect(endSpy).toHaveBeenCalledTimes(1);
        expect(logSpy).toHaveBeenCalled();
        expect(lane.__lnVectorDebugLogged.ln1).toBe(true);
    });

    it('respects explicit debug disable toggle', () => {
        globalThis.__LN_VECTOR_DEBUG = false;
        const lane = { laneIndex: 0 };
        const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
        logLayerNormVectorDump({ layerIndex: 0, kind: 'ln1', lane, vectors: {} });
        expect(logSpy).not.toHaveBeenCalled();
    });
});
