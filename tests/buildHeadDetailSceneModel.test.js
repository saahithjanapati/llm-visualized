import { describe, expect, it } from 'vitest';
import { D_MODEL } from '../src/ui/selectionPanelConstants.js';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import { buildTransformerSceneModel } from '../src/view2d/model/buildTransformerSceneModel.js';
import { flattenSceneNodes } from '../src/view2d/schema/sceneTypes.js';
import { resolveView2dCaptionLines } from '../src/view2d/captionUtils.js';

function createActivationSource(tokenCount = 4) {
    const promptTokens = Array.from({ length: tokenCount }, (_, index) => index);
    const tokenDisplayStrings = promptTokens.map((index) => `tok_${index}`);

    return {
        meta: {
            prompt_tokens: promptTokens,
            token_display_strings: tokenDisplayStrings
        },
        getLayerLn1(layerIndex, mode, tokenIndex) {
            if (mode !== 'shift') return null;
            return Array.from({ length: D_MODEL }, (_, index) => {
                const centered = (index % 11) - 5;
                return centered * (tokenIndex + 1) * 0.032;
            });
        }
    };
}

describe('buildHeadDetailSceneModel', () => {
    it('stores a captioned, wide head-detail scene alongside the selected head preview', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4),
            layerCount: 1,
            headDetailTarget: {
                layerIndex: 0,
                headIndex: 2
            }
        });

        const headDetailScene = scene.metadata.headDetailScene;
        const nodes = flattenSceneNodes(headDetailScene);
        const copyNodes = nodes.filter((node) => node.role === 'x-ln-copy');
        const connectorNodes = nodes.filter((node) => node.kind === 'connector');
        const firstCopy = copyNodes.find((node) => node.semantic?.branchKey === 'q');

        expect(headDetailScene).toBeTruthy();
        expect(copyNodes).toHaveLength(3);
        expect(connectorNodes).toHaveLength(3);
        expect(resolveView2dCaptionLines(firstCopy)).toEqual([
            { tex: 'X_{\\ln}^{Q}', text: 'X_ln^Q' },
            { tex: '4 \\\\times 768', text: '4 × 768' }
        ]);
        expect(firstCopy?.rowItems?.[0]?.semantic).toMatchObject({
            componentKind: 'residual',
            layerIndex: 0,
            headIndex: 2,
            stage: 'ln1.shift',
            branchKey: 'q',
            tokenIndex: 0
        });

        const layout = buildSceneLayout(headDetailScene);
        const firstEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'x-ln-copy'
            && entry.semantic?.branchKey === 'q'
        ));

        expect(firstEntry?.contentBounds.width).toBeGreaterThan(firstEntry?.contentBounds.height || 0);
    });
});
