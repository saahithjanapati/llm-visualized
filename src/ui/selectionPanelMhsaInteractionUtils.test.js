import { describe, expect, it } from 'vitest';

import {
    buildMhsaProjectionComponentStateKey,
    resolveMhsaKeyRowSceneFocus,
    resolveMhsaProjectionInputRowSceneFocus,
    resolveMhsaProjectionWeightSceneFocus,
    resolveMhsaValueInputRowSceneFocus,
    resolveMhsaValueRowSceneFocus,
    shouldMirrorMhsaHeadOutputRowFocus
} from './selectionPanelMhsaInteractionUtils.js';

describe('shouldMirrorMhsaHeadOutputRowFocus', () => {
    it('mirrors head-output row focus for query-row hovers on the Q stage', () => {
        expect(shouldMirrorMhsaHeadOutputRowFocus({
            hoverSource: 'query',
            projectionKind: 'q'
        })).toBe(true);
    });

    it('mirrors head-output row focus for X_ln row hovers on the Q stage', () => {
        expect(shouldMirrorMhsaHeadOutputRowFocus({
            hoverSource: 'x',
            projectionKind: 'q'
        })).toBe(true);
    });

    it('does not mirror focus for non-query hover sources', () => {
        expect(shouldMirrorMhsaHeadOutputRowFocus({
            hoverSource: 'value-post',
            projectionKind: 'q'
        })).toBe(false);
    });

    it('does not mirror focus for non-Q projection stages', () => {
        expect(shouldMirrorMhsaHeadOutputRowFocus({
            hoverSource: 'query',
            projectionKind: 'v'
        })).toBe(false);
    });
});

describe('resolveMhsaProjectionWeightSceneFocus', () => {
    it('maps W_Q hover to the first X_ln, local X_ln, weight, output, and query link', () => {
        expect(resolveMhsaProjectionWeightSceneFocus({
            projectionKind: 'q',
            stageIndex: 2,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(2, 'input'),
                buildMhsaProjectionComponentStateKey(2, 'weight'),
                buildMhsaProjectionComponentStateKey(2, 'output')
            ],
            attentionBlocks: ['query'],
            connectors: ['q']
        });
    });

    it('maps W_K hover to the first X_ln, local X_ln, weight, output, and K transpose link', () => {
        expect(resolveMhsaProjectionWeightSceneFocus({
            projectionKind: 'k',
            stageIndex: 1,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(1, 'input'),
                buildMhsaProjectionComponentStateKey(1, 'weight'),
                buildMhsaProjectionComponentStateKey(1, 'output')
            ],
            attentionBlocks: ['transpose'],
            connectors: ['k']
        });
    });

    it('deduplicates the first X_ln block when W_V belongs to the first stage', () => {
        expect(resolveMhsaProjectionWeightSceneFocus({
            projectionKind: 'v',
            stageIndex: 0,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(0, 'weight'),
                buildMhsaProjectionComponentStateKey(0, 'output')
            ],
            attentionBlocks: ['value-post'],
            connectors: ['v']
        });
    });
});

describe('resolveMhsaKeyRowSceneFocus', () => {
    it('keeps only the local K projection path active for K-row hovers', () => {
        expect(resolveMhsaKeyRowSceneFocus({
            keyStageIndex: 1,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(1, 'input'),
                buildMhsaProjectionComponentStateKey(1, 'multiply'),
                buildMhsaProjectionComponentStateKey(1, 'weight'),
                buildMhsaProjectionComponentStateKey(1, 'plus'),
                buildMhsaProjectionComponentStateKey(1, 'bias'),
                buildMhsaProjectionComponentStateKey(1, 'equals'),
                buildMhsaProjectionComponentStateKey(1, 'output')
            ],
            attentionBlocks: ['transpose'],
            connectors: ['k']
        });
    });

    it('deduplicates the copied X_ln input when K is the first stage', () => {
        expect(resolveMhsaKeyRowSceneFocus({
            keyStageIndex: 0,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(0, 'multiply'),
                buildMhsaProjectionComponentStateKey(0, 'weight'),
                buildMhsaProjectionComponentStateKey(0, 'plus'),
                buildMhsaProjectionComponentStateKey(0, 'bias'),
                buildMhsaProjectionComponentStateKey(0, 'equals'),
                buildMhsaProjectionComponentStateKey(0, 'output')
            ],
            attentionBlocks: ['transpose'],
            connectors: ['k']
        });
    });
});

describe('resolveMhsaProjectionInputRowSceneFocus', () => {
    it('keeps the first X_ln block plus the local projection path for copied Q input-row hovers', () => {
        expect(resolveMhsaProjectionInputRowSceneFocus({
            stageIndex: 0,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(0, 'multiply'),
                buildMhsaProjectionComponentStateKey(0, 'weight'),
                buildMhsaProjectionComponentStateKey(0, 'plus'),
                buildMhsaProjectionComponentStateKey(0, 'bias'),
                buildMhsaProjectionComponentStateKey(0, 'equals'),
                buildMhsaProjectionComponentStateKey(0, 'output')
            ]
        });
    });

    it('includes the first X_ln block when hovering a copied K/V input row', () => {
        expect(resolveMhsaProjectionInputRowSceneFocus({
            stageIndex: 2,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(2, 'input'),
                buildMhsaProjectionComponentStateKey(2, 'multiply'),
                buildMhsaProjectionComponentStateKey(2, 'weight'),
                buildMhsaProjectionComponentStateKey(2, 'plus'),
                buildMhsaProjectionComponentStateKey(2, 'bias'),
                buildMhsaProjectionComponentStateKey(2, 'equals'),
                buildMhsaProjectionComponentStateKey(2, 'output')
            ]
        });
    });
});

describe('resolveMhsaValueRowSceneFocus', () => {
    it('keeps the first X_ln block plus the local V path when hovering value rows', () => {
        expect(resolveMhsaValueRowSceneFocus({
            valueStageIndex: 2,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(2, 'input'),
                buildMhsaProjectionComponentStateKey(2, 'multiply'),
                buildMhsaProjectionComponentStateKey(2, 'weight'),
                buildMhsaProjectionComponentStateKey(2, 'plus'),
                buildMhsaProjectionComponentStateKey(2, 'bias'),
                buildMhsaProjectionComponentStateKey(2, 'equals'),
                buildMhsaProjectionComponentStateKey(2, 'output')
            ],
            attentionBlocks: ['value-post'],
            connectors: ['v']
        });
    });

    it('deduplicates the first X_ln block when the value stage is first', () => {
        expect(resolveMhsaValueRowSceneFocus({
            valueStageIndex: 0,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(0, 'multiply'),
                buildMhsaProjectionComponentStateKey(0, 'weight'),
                buildMhsaProjectionComponentStateKey(0, 'plus'),
                buildMhsaProjectionComponentStateKey(0, 'bias'),
                buildMhsaProjectionComponentStateKey(0, 'equals'),
                buildMhsaProjectionComponentStateKey(0, 'output')
            ],
            attentionBlocks: ['value-post'],
            connectors: ['v']
        });
    });
});

describe('resolveMhsaValueInputRowSceneFocus', () => {
    it('keeps the first X_ln block plus the local V projection path for V input-row hovers', () => {
        expect(resolveMhsaValueInputRowSceneFocus({
            valueStageIndex: 2,
            firstProjectionStageIndex: 0
        })).toEqual({
            projectionComponents: [
                buildMhsaProjectionComponentStateKey(0, 'input'),
                buildMhsaProjectionComponentStateKey(2, 'input'),
                buildMhsaProjectionComponentStateKey(2, 'multiply'),
                buildMhsaProjectionComponentStateKey(2, 'weight'),
                buildMhsaProjectionComponentStateKey(2, 'plus'),
                buildMhsaProjectionComponentStateKey(2, 'bias'),
                buildMhsaProjectionComponentStateKey(2, 'equals'),
                buildMhsaProjectionComponentStateKey(2, 'output')
            ]
        });
    });
});
