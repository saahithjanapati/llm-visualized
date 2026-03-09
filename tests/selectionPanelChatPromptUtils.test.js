import { describe, expect, it } from 'vitest';
import { buildSelectionChatPrompt } from '../src/ui/selectionPanelChatPromptUtils.js';

describe('selectionPanelChatPromptUtils', () => {
    it('adds shared-view instructions and self-attention color cues', () => {
        const prompt = buildSelectionChatPrompt({
            selection: {
                label: 'Query Vector',
                kind: 'vector',
                info: {
                    activationData: {
                        stage: 'qkv.q',
                        tokenIndex: 2,
                        tokenLabel: 'cat',
                        tokenId: 1337,
                        headIndex: 1,
                        layerIndex: 0
                    }
                }
            },
            normalizedLabel: 'Query Vector',
            title: 'Query Vector',
            promptContextSummary: 'the | [[cat]] | sat'
        });

        expect(prompt).toContain('watching the same animation alongside the user');
        expect(prompt).toContain('## Self-Attention Color Cues');
        expect(prompt).toContain('Query vectors are blue (#276ebb).');
        expect(prompt).toContain('Key vectors are green (#1e9f57).');
        expect(prompt).toContain('Value vectors are red (#aa3420).');
        expect(prompt).toContain('## Current Visualization State');
        expect(prompt).toContain('We are in the query-projection view for one attention head');
    });

    it('describes the live post-softmax attention state', () => {
        const prompt = buildSelectionChatPrompt({
            selection: {
                label: 'Attention Score',
                kind: 'matrix-cell',
                info: {
                    activationData: {
                        stage: 'attention.post',
                        tokenIndex: 2,
                        tokenLabel: 'cat',
                        keyTokenIndex: 1,
                        keyTokenLabel: 'sat',
                        headIndex: 0,
                        layerIndex: 0,
                        postScore: 0.8754321
                    }
                }
            },
            normalizedLabel: 'Attention Score'
        });

        expect(prompt).toContain('Attention mode: post-softmax');
        expect(prompt).toContain('The selected attention cell is currently being interpreted as a post-softmax weight.');
        expect(prompt).toContain('We are in the normalized attention step');
    });
});
