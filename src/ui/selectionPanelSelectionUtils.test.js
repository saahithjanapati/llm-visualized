import { describe, expect, it } from 'vitest';

import {
    isKvCacheVectorSelection,
    normalizeSelectionLabel,
    resolveSelectionLogitEntry,
    simplifyLayerNormParamDisplayLabel
} from './selectionPanelSelectionUtils.js';

describe('normalizeSelectionLabel', () => {
    it('renames LayerNorm normalized-stage selections to normalized residual stream vectors', () => {
        expect(normalizeSelectionLabel('LN1 Normed', {
            info: {
                activationData: {
                    stage: 'ln1.norm'
                }
            }
        })).toBe('Normalized Residual Stream Vector');

        expect(normalizeSelectionLabel('LN2 Normed', {
            info: {
                activationData: {
                    stage: 'ln2.norm'
                }
            }
        })).toBe('Normalized Residual Stream Vector');
    });

    it('renames LayerNorm product-stage selections to product vectors instead of parameter labels', () => {
        const selection = {
            info: {
                activationData: {
                    stage: 'ln1.scale',
                    layerNormKind: 'ln1'
                }
            }
        };

        expect(normalizeSelectionLabel('LayerNorm 1 Scale', selection)).toBe('LayerNorm 1 Product Vector');
        expect(simplifyLayerNormParamDisplayLabel('LayerNorm 1 Scale', selection)).toBe('LayerNorm 1 Product Vector');
    });

    it('treats explicit cached-KV metadata as a cached vector selection', () => {
        const selection = {
            label: 'Cached Key Vector',
            info: {
                cachedKv: true,
                activationData: {
                    label: 'Cached Key Vector',
                    stage: 'qkv.k',
                    cachedKv: true
                }
            }
        };

        expect(isKvCacheVectorSelection(selection)).toBe(true);
        expect(normalizeSelectionLabel('Key Vector', selection)).toBe('Cached Key Vector');
    });

    it('prefers nested logit entry metadata over the wrapper selection info object', () => {
        const nestedEntry = {
            token: 'hello',
            token_id: 31337,
            prob: 0.42
        };
        const selection = {
            label: 'Chosen token: hello',
            info: {
                tokenId: 31337,
                tokenIndex: 4,
                logitEntry: nestedEntry
            }
        };

        expect(resolveSelectionLogitEntry(selection)).toBe(nestedEntry);
    });
});
