import { describe, expect, it } from 'vitest';

import { buildSelectionChatPrompt } from './selectionPanelChatPromptUtils.js';

describe('buildSelectionChatPrompt', () => {
    it('includes explicit 2D surface context when the semantic canvas is active', () => {
        const prompt = buildSelectionChatPrompt({
            surfaceState: {
                activeMode: 'transformer-view2d',
                supports2d: true,
                supports3d: true,
                current2dFocusLabel: 'Layer 4 Attention Head 3',
                current2dDetailFocusLabel: 'Pre-Softmax Attention Score',
                current2dSemanticTarget: {
                    componentKind: 'mhsa',
                    layerIndex: 3,
                    headIndex: 2,
                    stage: 'attention',
                    role: 'attention-pre-score'
                },
                current2dTransitionMode: 'staged-head-detail',
                current2dSelectionSidebarVisible: true
            }
        });

        expect(prompt).toContain('## Question Context');
        expect(prompt).toContain('I am currently in the 2D semantic canvas.');
        expect(prompt).toContain('My current 2D focus label is: Layer 4 Attention Head 3');
        expect(prompt).toContain('My current 2D detail focus is: Pre-Softmax Attention Score');
        expect(prompt).toContain('My current 2D semantic target is: component mhsa, layer 4, head 3, stage attention, role attention-pre-score');
        expect(prompt).toContain('This 2D attention-head detail view reads left to right: X_ln is copied into query, key, and value branches before the head-specific attention computation continues to the right.');
        expect(prompt).toContain('The matrices are shown as rounded matrix cards, and token-indexed matrices such as X_ln, Q, K, V, A_pre, A_post, and H_i can be inspected row by row by hovering their rows.');
        expect(prompt).toContain('Each branch starts from a T x 768 input matrix and applies the head-specific W_Q, W_K, or W_V matrix with shape 768 x 64 plus the corresponding 1 x 64 bias vector to produce Q, K, and V with shape T x 64.');
        expect(prompt).toContain('The query and key branches combine into Q K^T / sqrt(d_head) to form the T x T pre-softmax matrix A_pre, then masking and softmax produce A_post, and A_post multiplies V to produce the head output H_i with shape T x 64.');
        expect(prompt).toContain('The 2D selection sidebar is currently visible: yes');
    });

    it('frames the copied text as background for the user question and keeps 3D mode explicit', () => {
        const prompt = buildSelectionChatPrompt({
            surfaceState: {
                activeMode: '3d',
                supports2d: true,
                supports3d: true,
                available2dFocusLabel: 'Layer 1 Self-Attention',
                available2dDetailFocusLabel: 'Query Weight Matrix'
            },
            liveSceneState: {
                currentLayerIndex: 4,
                displayStage: 'Multi-Head Self-Attention',
                detailStageKey: 'qkv_projection',
                detailStageLabel: 'Q/K/V projections',
                isFinalStage: false,
                forwardPassComplete: false
            }
        });

        expect(prompt).toContain('I am about to ask a question about this GPT-2 visualization.');
        expect(prompt).toContain('I am currently in the 3D scene.');
        expect(prompt).toContain('If I open the matching 2D target for my current selection, it would focus on: Layer 1 Self-Attention');
        expect(prompt).toContain('If I open this in 2D, the detailed focus would be: Query Weight Matrix');
        expect(prompt).toContain('## Current Live Scene');
        expect(prompt).toContain('The live scene is currently working in Layer 5.');
        expect(prompt).toContain('The live scene is currently in: Multi-Head Self-Attention');
        expect(prompt).toContain('A more specific live substage appears to be: Q/K/V projections');
        expect(prompt).toContain('## My Question');
        expect(prompt).toContain('INSERT YOUR QUESTION HERE');
    });
});
