import { describe, expect, it } from 'vitest';

import { buildTransformerSceneModel } from './model/buildTransformerSceneModel.js';
import { flattenSceneNodes, VIEW2D_NODE_KINDS } from './schema/sceneTypes.js';
import {
    buildResidualOverviewSelectionLockState,
    buildResidualRowHoverPayload,
    buildResidualRowSelectionFocusState,
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
            },
            detailSemanticTargets: [{
                componentKind: 'layer-norm',
                layerIndex: 2,
                stage: 'ln1.param.scale',
                role: 'layer-norm-scale'
            }],
            detailFocusLabel: 'LayerNorm 1 Scale'
        });
        expect(context?.transitionMode || '').toBe('direct');
    });

    it('maps token selections onto the overview chip target that exists in the 2D scene', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Token: Gamma',
            info: {
                tokenIndex: 2,
                positionIndex: 3,
                activationData: {
                    stage: 'embedding.token',
                    tokenIndex: 2,
                    positionIndex: 3
                }
            }
        }, 'Token: Gamma');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'input-token-chip-group',
                tokenIndex: 2,
                positionIndex: 3
            }
        });
        expect(context?.transitionMode || '').toBe('staged-focus');
    });

    it('maps vocabulary embedding selections to the overview embedding card instead of a synthetic stage', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Vocabulary Embedding Matrix',
            info: {}
        }, 'Vocabulary Embedding Matrix');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'module'
            },
            focusLabel: 'Token embeddings'
        });
    });

    it('includes an initial overview row lock target for residual vector selections', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Residual Stream Vector',
            info: {
                layerIndex: 2,
                tokenIndex: 1,
                tokenLabel: 'Token B',
                activationData: {
                    stage: 'layer.incoming',
                    layerIndex: 2,
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        }, 'Residual Stream Vector');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 2,
                stage: 'incoming',
                role: 'module'
            },
            initialOverviewSelectionLockTarget: {
                semanticTarget: {
                    componentKind: 'residual',
                    layerIndex: 2,
                    stage: 'incoming',
                    role: 'module'
                },
                tokenIndex: 1,
                tokenLabel: 'Token B'
            }
        });
    });

    it('maps embedding-sum vector selections to the incoming residual overview row', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Embedding Sum - Token B',
            info: {
                layerIndex: 0,
                tokenIndex: 1,
                tokenLabel: 'Token B',
                activationData: {
                    stage: 'embedding.sum',
                    layerIndex: 0,
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        }, 'Residual Stream Vector');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module'
            },
            initialOverviewSelectionLockTarget: {
                semanticTarget: {
                    componentKind: 'residual',
                    layerIndex: 0,
                    stage: 'incoming',
                    role: 'module'
                },
                tokenIndex: 1,
                tokenLabel: 'Token B'
            }
        });
        expect(context?.transitionMode || '').toBe('staged-focus');
    });

    it('maps intermediate post-MLP residual selections to the next layer incoming residual overview row', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Post-MLP Residual - Token B',
            info: {
                layerIndex: 4,
                tokenIndex: 1,
                tokenLabel: 'Token B',
                activationData: {
                    stage: 'residual.post_mlp',
                    layerIndex: 4,
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        }, 'Residual Stream Vector');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 5,
                stage: 'incoming',
                role: 'module'
            },
            initialOverviewSelectionLockTarget: {
                semanticTarget: {
                    componentKind: 'residual',
                    layerIndex: 5,
                    stage: 'incoming',
                    role: 'module'
                },
                tokenIndex: 1,
                tokenLabel: 'Token B'
            }
        });
        expect(context?.transitionMode || '').toBe('staged-focus');
    });

    it('maps final post-MLP residual selections to the outgoing residual overview row', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Post-MLP Residual - Token B',
            info: {
                layerIndex: 11,
                tokenIndex: 1,
                tokenLabel: 'Token B',
                activationData: {
                    stage: 'residual.post_mlp',
                    layerIndex: 11,
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        }, 'Residual Stream Vector');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 11,
                stage: 'outgoing',
                role: 'module'
            },
            initialOverviewSelectionLockTarget: {
                semanticTarget: {
                    componentKind: 'residual',
                    layerIndex: 11,
                    stage: 'outgoing',
                    role: 'module'
                },
                tokenIndex: 1,
                tokenLabel: 'Token B'
            }
        });
        expect(context?.transitionMode || '').toBe('staged-focus');
    });

    it('builds a row-scoped overview focus state for residual selections', () => {
        const scene = buildTransformerSceneModel({
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1
        });
        const nodes = flattenSceneNodes(scene);
        const residualNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
        ));
        const residualNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'incoming'
        ));

        const focusResult = buildResidualRowSelectionFocusState(scene, {
            node: residualNode,
            rowHit: {
                rowIndex: 1,
                rowItem: residualNode?.rowItems?.[1]
            }
        });

        expect(focusResult?.focusState?.activeNodeIds).toContain(residualNode?.id);
        expect(focusResult?.focusState?.rowSelections).toEqual(expect.arrayContaining(
            residualNodes.map((node) => ({
                nodeId: node.id,
                rowIndex: 1
            }))
        ));
        expect(focusResult?.focusState?.rowSelections).toHaveLength(residualNodes.length);
        expect(focusResult?.focusState?.activeNodeIds).toEqual(expect.arrayContaining(
            residualNodes.map((node) => node.id)
        ));
        expect(focusResult?.focusState?.activeConnectorIds?.length || 0).toBeGreaterThan(0);
    });

    it('resolves a residual overview row lock state from a normalized selection target', () => {
        const scene = buildTransformerSceneModel({
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1
        });

        const lockState = buildResidualOverviewSelectionLockState(scene, {
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module'
            },
            tokenIndex: 1,
            tokenLabel: 'Token B'
        }, {
            getTokenId(tokenIndex = 0) {
                return tokenIndex + 100;
            }
        });

        expect(lockState?.label).toBe('Residual Stream Vector');
        expect(lockState?.info?.tokenIndex).toBe(1);
        expect(lockState?.info?.tokenId).toBe(101);
        expect(lockState?.focusState?.rowSelections).toEqual(expect.arrayContaining([
            expect.objectContaining({
                rowIndex: 1
            })
        ]));
    });

    it('opens attention head targets directly into the head-detail scene', () => {
        expect(resolveTransformerView2dOpenTransitionMode({
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 3,
                stage: 'attention',
                role: 'head'
            }
        })).toBe('direct');
    });

    it('maps attention-score selections to a direct head-detail open', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Post-Softmax Attention Score',
            info: {
                layerIndex: 1,
                headIndex: 2,
                activationData: {
                    stage: 'attention.post',
                    layerIndex: 1,
                    headIndex: 2
                }
            }
        }, 'Post-Softmax Attention Score');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            },
            detailSemanticTargets: [{
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'attention-post'
            }],
            detailFocusLabel: 'Post-Softmax Attention Score'
        });
        expect(context?.transitionMode || '').toBe('direct');
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

    it('prefers the actual residual matrix card for post-MLP overview focus targets', () => {
        expect(resolveFocusSemanticTargets({
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 4,
                stage: 'post-mlp-residual',
                role: 'module'
            }
        })).toEqual([
            {
                componentKind: 'residual',
                layerIndex: 5,
                stage: 'incoming',
                role: 'module-card'
            },
            {
                componentKind: 'residual',
                layerIndex: 5,
                stage: 'incoming',
                role: 'module'
            }
        ]);
    });

    it('uses the explicit direct-open mode for non-head detail routes', () => {
        expect(resolveTransformerView2dOpenTransitionMode({
            semanticTarget: {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'projection-weight'
            }
        })).toBe('direct');
    });

    it('opens layer norm targets on the direct scene-backed detail path', () => {
        expect(resolveTransformerView2dOpenTransitionMode({
            semanticTarget: {
                componentKind: 'layer-norm',
                layerIndex: 5,
                stage: 'ln2',
                role: 'module'
            }
        })).toBe('direct');
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
            detailSemanticTargets: [{
                componentKind: 'output-projection',
                layerIndex: 4,
                stage: 'attn-out',
                role: 'concat-output-copy-matrix'
            }],
            detailFocusLabel: 'Concatenate Heads',
            focusLabel: 'Layer 5 Output Projection'
        });
        expect(context?.transitionMode || '').toBe('direct');
    });

    it('maps output-projection bias selections to a direct detail open with bias highlighting', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Output Projection Bias Vector',
            info: {
                layerIndex: 4,
                activationData: {
                    stage: 'attention.output_projection.bias',
                    layerIndex: 4
                }
            }
        }, 'Output Projection Bias Vector');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'output-projection',
                layerIndex: 4,
                stage: 'attn-out',
                role: 'projection-weight'
            },
            detailSemanticTargets: [{
                componentKind: 'output-projection',
                layerIndex: 4,
                stage: 'attn-out',
                role: 'projection-bias'
            }],
            detailFocusLabel: 'Output Projection Bias Vector'
        });
        expect(context?.transitionMode || '').toBe('direct');
    });

    it('maps MLP output selections to the overview module and detail-row highlight target', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'MLP Down Projection',
            info: {
                layerIndex: 4,
                tokenIndex: 1,
                activationData: {
                    stage: 'mlp.down',
                    layerIndex: 4,
                    tokenIndex: 1
                }
            }
        }, 'MLP Down Projection');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'mlp',
                layerIndex: 4,
                stage: 'mlp',
                role: 'module'
            },
            detailSemanticTargets: [{
                componentKind: 'mlp',
                layerIndex: 4,
                stage: 'mlp-down',
                role: 'mlp-down-output'
            }],
            detailFocusLabel: 'MLP Down Projection'
        });
        expect(context?.transitionMode || '').toBe('direct');
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

    it('exposes hover context for overview residual modules when row hits are unavailable', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'module-card',
                semantic: {
                    componentKind: 'residual',
                    layerIndex: 0,
                    stage: 'incoming',
                    role: 'module-card'
                }
            }
        });

        expect(payload).toEqual({
            label: 'Residual Stream Vector',
            info: {
                layerIndex: 0,
                suppressTokenChip: true,
                activationData: {
                    label: 'Residual Stream Vector',
                    stage: 'layer.incoming',
                    layerIndex: 0,
                    suppressTokenChip: true
                }
            }
        });
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
            label: 'Token: Gamma',
            info: {
                tokenIndex: 2,
                tokenLabel: 'Gamma',
                positionIndex: 3,
                activationData: {
                    label: 'Token: Gamma',
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
                    tokenLabel: 'Gamma',
                    positionIndex: 3
                }
            }
        });

        expect(payload).toEqual({
            label: 'Position: 3',
            info: {
                tokenIndex: 2,
                tokenLabel: 'Gamma',
                positionIndex: 3,
                activationData: {
                    label: 'Position: 3',
                    stage: 'embedding.position',
                    tokenIndex: 2,
                    tokenLabel: 'Gamma',
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
            label: 'Chosen Token: Gamma',
            info: {
                tokenIndex: 2,
                tokenLabel: 'Gamma',
                positionIndex: 3,
                activationData: {
                    label: 'Chosen Token: Gamma',
                    stage: 'generation.chosen',
                    tokenIndex: 2,
                    tokenLabel: 'Gamma',
                    positionIndex: 3
                }
            }
        });
    });

    it('exposes hover context for the overview vocabulary embedding matrix', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'vocabulary-embedding-card',
                semantic: {
                    componentKind: 'embedding',
                    stage: 'embedding.token',
                    role: 'vocabulary-embedding-card'
                }
            }
        });

        expect(payload).toEqual({
            label: 'Vocabulary Embedding Matrix',
            info: {
                suppressTokenChip: true,
                activationData: {
                    label: 'Vocabulary Embedding Matrix',
                    stage: 'embedding.token',
                    suppressTokenChip: true
                }
            }
        });
    });

    it('exposes hover context for the overview position embedding matrix', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'position-embedding-card',
                semantic: {
                    componentKind: 'embedding',
                    stage: 'embedding.position',
                    role: 'position-embedding-card'
                }
            }
        });

        expect(payload).toEqual({
            label: 'Position Embedding Matrix',
            info: {
                suppressTokenChip: true,
                activationData: {
                    label: 'Position Embedding Matrix',
                    stage: 'embedding.position',
                    suppressTokenChip: true
                }
            }
        });
    });

    it('exposes hover context for the overview unembedding matrix', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'unembedding',
                semantic: {
                    componentKind: 'logits',
                    stage: 'unembedding',
                    role: 'unembedding'
                }
            }
        });

        expect(payload).toEqual({
            label: 'Vocabulary Unembedding Matrix',
            info: {
                suppressTokenChip: true,
                activationData: {
                    label: 'Vocabulary Unembedding Matrix',
                    stage: 'unembedding',
                    suppressTokenChip: true
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

    it('exposes hover context for overview layer norm modules with the module label', () => {
        const payload = buildSemanticNodeHoverPayload({
            entry: {
                role: 'module-title',
                semantic: {
                    componentKind: 'layer-norm',
                    layerIndex: 5,
                    stage: 'ln2',
                    role: 'module-title'
                }
            }
        });

        expect(payload).toEqual({
            label: 'LayerNorm 2',
            info: {
                layerIndex: 5,
                layerNormKind: 'ln2',
                suppressTokenChip: true,
                activationData: {
                    label: 'LayerNorm 2',
                    stage: 'ln2',
                    layerIndex: 5,
                    layerNormKind: 'ln2',
                    suppressTokenChip: true
                }
            }
        });
    });

    it('builds overview component focus state for hovered residual modules', () => {
        const scene = buildTransformerSceneModel({
            layerCount: 1
        });
        const nodes = flattenSceneNodes(scene);
        const incomingResidualNode = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.MATRIX
            && node.semantic?.componentKind === 'residual'
            && node.semantic?.stage === 'incoming'
            && node.role === 'module-card'
        ));
        const ln1CardNode = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.MATRIX
            && node.semantic?.componentKind === 'layer-norm'
            && node.semantic?.stage === 'ln1'
            && node.role === 'module-card'
        ));
        const residualToLn1Connector = nodes.find((node) => (
            node.kind === VIEW2D_NODE_KINDS.CONNECTOR
            && node.source?.nodeId === incomingResidualNode?.id
            && node.target?.nodeId === ln1CardNode?.id
        ));

        const hoverState = buildSemanticNodeHoverFocusState(scene, {
            entry: incomingResidualNode
        });

        expect(hoverState?.focusState?.activeNodeIds).toContain(incomingResidualNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(residualToLn1Connector?.id);
        expect(Array.isArray(hoverState?.focusState?.activeNodeIds)).toBe(true);
        expect(typeof hoverState?.signature).toBe('string');
        expect(hoverState?.signature.length).toBeGreaterThan(0);
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
