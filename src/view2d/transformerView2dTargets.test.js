import { describe, expect, it } from 'vitest';

import { buildTransformerSceneModel } from './model/buildTransformerSceneModel.js';
import { flattenSceneNodes, VIEW2D_NODE_KINDS } from './schema/sceneTypes.js';
import {
    buildResidualRowHoverPayload,
    buildSemanticNodeHoverFocusState,
    buildSemanticNodeHoverPayload,
    isLayerNormOverviewEntry,
    isMlpOverviewEntry,
    isOutputProjectionOverviewEntry,
    TRANSFORMER_VIEW2D_OVERVIEW_LABEL,
    resolveDetailTargetsFromSemanticTarget,
    resolveFocusSemanticTargets,
    resolveTransformerView2dActionContext,
    resolveTransformerView2dOpenTransitionMode,
    resolveTransformerView2dStageHeader
} from './transformerView2dTargets.js';

describe('transformerView2dTargets', () => {
    it('maps layer norm selections onto the scene-backed layer norm detail target', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'LayerNorm 1 Scale',
            info: {
                layerIndex: 2,
                activationData: {
                    stage: 'ln1.param.scale',
                    layerIndex: 2
                }
            }
        }, 'LayerNorm 1 Scale');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'layer-norm',
                layerIndex: 2,
                stage: 'ln1',
                role: 'module'
            }
        });
        expect(context?.transitionMode || '').toBe('');
    });

    it('uses the staged head-detail opening flow for attention head targets', () => {
        expect(resolveTransformerView2dOpenTransitionMode({
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 3,
                stage: 'attention',
                role: 'head'
            }
        })).toBe('staged-head-detail');
    });

    it('prefers the layer norm card and title when resolving focus candidates', () => {
        expect(resolveFocusSemanticTargets({
            semanticTarget: {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln2',
                role: 'module'
            }
        })).toEqual([
            {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln2',
                role: 'module-card'
            },
            {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln2',
                role: 'module-title'
            },
            {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln2',
                role: 'module'
            }
        ]);
    });

    it('leaves non-head detail routes on the existing direct-open path', () => {
        expect(resolveTransformerView2dOpenTransitionMode({
            semanticTarget: {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'projection-weight'
            }
        })).toBe('');
    });

    it('opens layer norm targets on the direct scene-backed detail path', () => {
        expect(resolveTransformerView2dOpenTransitionMode({
            semanticTarget: {
                componentKind: 'layer-norm',
                layerIndex: 5,
                stage: 'ln2',
                role: 'module'
            }
        })).toBe('');
    });

    it('redirects concat overview targets into output-projection detail', () => {
        expect(resolveDetailTargetsFromSemanticTarget({
            componentKind: 'mhsa',
            layerIndex: 2,
            stage: 'concatenate',
            role: 'concat'
        })).toEqual({
            headDetailTarget: null,
            concatDetailTarget: null,
            outputProjectionDetailTarget: {
                layerIndex: 2
            },
            mlpDetailTarget: null,
            layerNormDetailTarget: null
        });
    });

    it('routes layer norm semantic targets into the dedicated layer norm detail scene', () => {
        expect(resolveDetailTargetsFromSemanticTarget({
            componentKind: 'layer-norm',
            layerIndex: 4,
            stage: 'ln2',
            role: 'module'
        })).toEqual({
            headDetailTarget: null,
            concatDetailTarget: null,
            outputProjectionDetailTarget: null,
            mlpDetailTarget: null,
            layerNormDetailTarget: {
                layerNormKind: 'ln2',
                layerIndex: 4
            }
        });
    });

    it('maps concatenate selections to output projection for 2D view actions', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Concatenate Heads',
            info: {
                layerIndex: 4,
                activationData: {
                    stage: 'attention.concatenate',
                    layerIndex: 4
                }
            }
        }, 'Concatenate Heads');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'output-projection',
                layerIndex: 4,
                stage: 'attn-out',
                role: 'projection-weight'
            },
            focusLabel: 'Layer 5 Output Projection'
        });
    });

    it('formats the top-left stage header for layer stages', () => {
        expect(resolveTransformerView2dStageHeader({
            componentKind: 'layer-norm',
            layerIndex: 3,
            stage: 'ln1',
            role: 'module'
        })).toEqual({
            layerLabel: 'Layer 4',
            stageLabel: 'LayerNorm 1',
            fullLabel: 'Layer 4 LayerNorm 1'
        });
    });

    it('treats layer norm overview cards and titles as overview entries', () => {
        expect(isLayerNormOverviewEntry({
            role: 'module-card',
            semantic: {
                componentKind: 'layer-norm',
                layerIndex: 1,
                stage: 'ln2'
            }
        })).toBe(true);
        expect(isLayerNormOverviewEntry({
            role: 'module-title',
            semantic: {
                componentKind: 'layer-norm',
                stage: 'final-ln'
            }
        })).toBe(true);
    });

    it('formats the top-left stage header for attention heads', () => {
        expect(resolveTransformerView2dStageHeader({
            componentKind: 'mhsa',
            layerIndex: 7,
            headIndex: 5,
            stage: 'attention',
            role: 'head'
        })).toEqual({
            layerLabel: 'Layer 8',
            stageLabel: 'Attention Head 6',
            fullLabel: 'Layer 8 Attention Head 6'
        });
    });

    it('collapses MLP detail variants into a single stage header label', () => {
        expect(resolveTransformerView2dStageHeader({
            componentKind: 'mlp',
            layerIndex: 1,
            stage: 'mlp-up',
            role: 'mlp-up'
        })).toEqual({
            layerLabel: 'Layer 2',
            stageLabel: 'Multilayer Perceptron',
            fullLabel: 'Layer 2 Multilayer Perceptron'
        });
    });

    it('falls back to the overview header when no semantic target is active', () => {
        expect(resolveTransformerView2dStageHeader()).toEqual({
            layerLabel: '',
            stageLabel: TRANSFORMER_VIEW2D_OVERVIEW_LABEL,
            fullLabel: TRANSFORMER_VIEW2D_OVERVIEW_LABEL
        });
    });

    it('treats both output-projection title lines as overview entries', () => {
        expect(isOutputProjectionOverviewEntry({
            role: 'module-title-top',
            semantic: {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'module-title-top'
            }
        })).toBe(true);
        expect(isOutputProjectionOverviewEntry({
            role: 'module-title-bottom',
            semantic: {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'module-title-bottom'
            }
        })).toBe(true);
    });

    it('treats both MLP title lines as overview entries', () => {
        expect(isMlpOverviewEntry({
            role: 'module-title-top',
            semantic: {
                componentKind: 'mlp',
                layerIndex: 5,
                stage: 'mlp',
                role: 'module-title-top'
            }
        })).toBe(true);
        expect(isMlpOverviewEntry({
            role: 'module-title-bottom',
            semantic: {
                componentKind: 'mlp',
                layerIndex: 5,
                stage: 'mlp',
                role: 'module-title-bottom'
            }
        })).toBe(true);
    });

    it('formats post-layernorm residual row hover labels from the semantic stage', () => {
        const payload = buildResidualRowHoverPayload({
            rowItem: {
                label: 'Token A',
                semantic: {
                    componentKind: 'residual',
                    layerIndex: 0,
                    stage: 'ln2.output',
                    tokenIndex: 0
                }
            }
        }, {
            getTokenId() {
                return 42;
            }
        });

        expect(payload?.label).toBe('Post LayerNorm 2 Residual Vector');
        expect(payload?.info?.activationData?.label).toBe('Post LayerNorm 2 Residual Vector');
        expect(payload?.info?.activationData?.stage).toBe('ln2.output');
        expect(payload?.info?.activationData?.sourceStage).toBe('ln2.shift');
    });

    it('exposes hover token context for input token chips', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'input-token-chip',
                semantic: {
                    componentKind: 'embedding',
                    stage: 'embedding.token',
                    role: 'input-token-chip',
                    tokenIndex: 2
                },
                metadata: {
                    tokenLabel: 'Gamma'
                }
            }
        });

        expect(payload).toEqual({
            label: 'Input Token',
            info: {
                tokenIndex: 2,
                tokenLabel: 'Gamma',
                positionIndex: 3,
                activationData: {
                    label: 'Input Token',
                    stage: 'embedding.token',
                    tokenIndex: 2,
                    tokenLabel: 'Gamma',
                    positionIndex: 3
                }
            }
        });
    });

    it('exposes hover token context for input position chips', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'input-position-chip',
                semantic: {
                    componentKind: 'embedding',
                    stage: 'embedding.position',
                    role: 'input-position-chip',
                    tokenIndex: 2,
                    positionIndex: 3
                },
                metadata: {
                    tokenLabel: '3',
                    positionIndex: 3
                }
            }
        });

        expect(payload).toEqual({
            label: 'Input Position',
            info: {
                tokenIndex: 2,
                tokenLabel: '3',
                positionIndex: 3,
                activationData: {
                    label: 'Input Position',
                    stage: 'embedding.position',
                    tokenIndex: 2,
                    tokenLabel: '3',
                    positionIndex: 3
                }
            }
        });
    });

    it('exposes hover token context for chosen token chips', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'chosen-token-chip',
                semantic: {
                    componentKind: 'logits',
                    stage: 'output',
                    role: 'chosen-token-chip',
                    tokenIndex: 2
                },
                metadata: {
                    tokenLabel: 'Gamma'
                }
            }
        });

        expect(payload).toEqual({
            label: 'Chosen Token',
            info: {
                tokenIndex: 2,
                tokenLabel: 'Gamma',
                positionIndex: 3,
                activationData: {
                    label: 'Chosen Token',
                    stage: 'generation.chosen',
                    tokenIndex: 2,
                    tokenLabel: 'Gamma',
                    positionIndex: 3
                }
            }
        });
    });

    it('exposes hover context for overview attention heads', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'head-card',
                semantic: {
                    componentKind: 'mhsa',
                    layerIndex: 1,
                    headIndex: 2,
                    stage: 'attention',
                    role: 'head-card'
                }
            }
        });

        expect(payload).toEqual({
            label: 'Attention Head 3',
            info: {
                layerIndex: 1,
                headIndex: 2,
                suppressTokenChip: true,
                activationData: {
                    label: 'Attention Head 3',
                    stage: 'attention.head',
                    layerIndex: 1,
                    headIndex: 2,
                    suppressTokenChip: true
                }
            }
        });
    });

    it('exposes hover context for overview MLP modules', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'module-title-top',
                semantic: {
                    componentKind: 'mlp',
                    layerIndex: 5,
                    stage: 'mlp',
                    role: 'module-title-top'
                }
            }
        });

        expect(payload).toEqual({
            label: 'Multilayer Perceptron',
            info: {
                layerIndex: 5,
                suppressTokenChip: true,
                activationData: {
                    label: 'Multilayer Perceptron',
                    stage: 'mlp',
                    layerIndex: 5,
                    suppressTokenChip: true
                }
            }
        });
    });

    it('builds overview component focus state for hovered MLP modules', () => {
        const scene = buildTransformerSceneModel({
            layerCount: 1
        });
        const nodes = flattenSceneNodes(scene);
        const mlpCardNode = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.MATRIX
            && node.semantic?.componentKind === 'mlp'
            && node.semantic?.stage === 'mlp'
            && node.role === 'module-card'
        ));
        const mlpTitleTopNode = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.TEXT
            && node.semantic?.componentKind === 'mlp'
            && node.semantic?.stage === 'mlp'
            && node.role === 'module-title-top'
        ));
        const mlpTitleBottomNode = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.TEXT
            && node.semantic?.componentKind === 'mlp'
            && node.semantic?.stage === 'mlp'
            && node.role === 'module-title-bottom'
        ));
        const ln2CardNode = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.MATRIX
            && node.semantic?.componentKind === 'layer-norm'
            && node.semantic?.stage === 'ln2'
            && node.role === 'module-card'
        ));
        const postMlpAddNode = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.MATRIX
            && node.semantic?.componentKind === 'residual'
            && node.semantic?.stage === 'post-mlp-add'
            && node.role === 'add-circle'
        ));
        const ln2ToMlpConnector = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.CONNECTOR
            && node.source?.nodeId === ln2CardNode?.id
            && node.target?.nodeId === mlpCardNode?.id
        ));
        const mlpToAddConnector = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.CONNECTOR
            && node.source?.nodeId === mlpCardNode?.id
            && node.target?.nodeId === postMlpAddNode?.id
        ));

        const hoverState = buildSemanticNodeHoverFocusState(scene, {
            entry: mlpCardNode
        });

        expect(hoverState?.focusState?.activeNodeIds).toContain(mlpCardNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(mlpTitleTopNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(mlpTitleBottomNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(ln2ToMlpConnector?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(mlpToAddConnector?.id);
        expect(Array.isArray(hoverState?.focusState?.activeNodeIds)).toBe(true);
        expect(typeof hoverState?.signature).toBe('string');
        expect(hoverState?.signature.length).toBeGreaterThan(0);
    });
});
