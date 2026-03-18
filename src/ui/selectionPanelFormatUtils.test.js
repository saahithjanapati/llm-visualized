import { describe, expect, it } from 'vitest';

import { formatActivationData } from './selectionPanelFormatUtils.js';

describe('selectionPanelFormatUtils', () => {
    it('uses query-token terminology for attention-score activation data', () => {
        const text = formatActivationData({
            stage: 'attention.pre',
            layerIndex: 1,
            tokenIndex: 2,
            tokenLabel: 'cat',
            keyTokenIndex: 4,
            keyTokenLabel: 'sat'
        });

        expect(text).toContain('Query token: 3 (cat)');
        expect(text).toContain('Source token: 5 (sat)');
        expect(text).not.toContain('Target token:');
    });
});
