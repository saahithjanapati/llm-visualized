import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';
import { CaptureActivationSource } from '../src/data/CaptureActivationSource.js';
import { applyAttentionOutputProjectionDataToVector, applyPostAttentionResidualDataToVector } from '../src/animations/mhsa/mhsaActivationVectorUtils.js';
import { SelfAttentionAnimator } from '../src/animations/mhsa/SelfAttentionAnimator.js';
import { applyQkvProcessedVisuals } from '../src/animations/mhsa/VectorMatrixPassThrough.js';
import { applyVectorData } from '../src/engine/layers/gpt2LayerUtils.js';
import { recolorVectorFromData } from '../src/engine/layerPipelineMath.js';
import { sliceMlpActivationIntoSegments } from '../src/engine/layers/gpt2MlpActivationSegments.js';

const capturePath = process.env.ACTIVATION_CAPTURE_PATH
    ? path.resolve(process.env.ACTIVATION_CAPTURE_PATH)
    : null;
const describeIfCapture = capturePath ? describe : describe.skip;
const capture = capturePath
    ? JSON.parse(fs.readFileSync(capturePath, 'utf8'))
    : null;
const source = capture ? new CaptureActivationSource(capture) : null;
const activations = capture?.activations || {};
const layers = Array.isArray(activations.layers) ? activations.layers : [];
const tokenCount = source?.getTokenCount?.() || 0;
const baseVectorLength = source?.getBaseVectorLength?.() || 0;
const headCount = Array.isArray(layers[0]?.qkv?.q) ? layers[0].qkv.q.length : 0;

function cleanNumber(value) {
    return Number.isFinite(value) ? value : 0;
}

function decodeVectorEntry(entry) {
    if (Array.isArray(entry)) return entry.map(cleanNumber);
    if (entry && typeof entry === 'object' && Array.isArray(entry.v)) {
        const scale = Number.isFinite(entry.s) ? entry.s : 1;
        return entry.v.map((value) => cleanNumber(value) * scale);
    }
    return [];
}

function decodePackedAttentionRow(headEntry, rowIndex) {
    if (Array.isArray(headEntry)) {
        return decodeVectorEntry(headEntry[rowIndex]);
    }
    if (!headEntry || typeof headEntry !== 'object' || !Array.isArray(headEntry.v)) {
        return null;
    }

    const tokenCount = Number.isFinite(headEntry.n) ? Math.floor(headEntry.n) : 0;
    const rowStart = (rowIndex * (rowIndex + 1)) / 2;
    const rowEnd = rowStart + rowIndex + 1;
    const lower = headEntry.v
        .slice(rowStart, rowEnd)
        .map((value) => cleanNumber(value) * (Number.isFinite(headEntry.s) ? headEntry.s : 1));
    if (Array.isArray(headEntry.rs) && Number.isFinite(headEntry.rs[rowIndex])) {
        const rowScale = headEntry.rs[rowIndex];
        for (let index = 0; index < lower.length; index += 1) lower[index] *= rowScale;
    }

    if (!Array.isArray(headEntry.u) || tokenCount <= rowIndex + 1) {
        return lower;
    }

    const upperLength = tokenCount - rowIndex - 1;
    const upperStart = (rowIndex * ((2 * tokenCount) - rowIndex - 1)) / 2;
    const upperEnd = upperStart + upperLength;
    const upper = headEntry.u
        .slice(upperStart, upperEnd)
        .map((value) => cleanNumber(value) * (Number.isFinite(headEntry.us) ? headEntry.us : 1));
    if (Array.isArray(headEntry.urs) && Number.isFinite(headEntry.urs[rowIndex])) {
        const rowScale = headEntry.urs[rowIndex];
        for (let index = 0; index < upper.length; index += 1) upper[index] *= rowScale;
    }
    return lower.concat(upper);
}

function assertNumberClose(actual, expected, context, epsilon = 1e-3) {
    expect(Math.abs(actual - expected), context).toBeLessThanOrEqual(epsilon);
}

function assertArrayClose(actual, expected, context, epsilon = 1e-3) {
    expect(Array.isArray(actual), `${context}: expected array`).toBe(true);
    expect(actual.length, `${context}: length mismatch`).toBe(expected.length);
    for (let index = 0; index < expected.length; index += 1) {
        assertNumberClose(actual[index], expected[index], `${context}: index ${index}`, epsilon);
    }
}

function createVectorDouble({ rawData = [], instanceCount = null, processed = false } = {}) {
    const vector = {
        rawData: rawData.slice(),
        instanceCount: Number.isFinite(instanceCount) ? instanceCount : Math.max(1, rawData.length || 1),
        userData: {},
        group: {
            userData: {},
            position: new THREE.Vector3(),
            visible: true
        },
        mesh: {
            userData: {}
        },
        updateKeyColorsFromData: vi.fn(),
        setUniformColor: vi.fn()
    };

    if (processed) {
        vector.applyProcessedVisuals = vi.fn((data, numVisibleOutputUnits, colorOptions, visualOptions, cacheKeyData) => {
            vector.rawData = Array.from(data);
            vector.__processed = {
                data: Array.from(data),
                numVisibleOutputUnits,
                colorOptions,
                visualOptions,
                cacheKeyData
            };
        });
    }

    return vector;
}

describeIfCapture('activation pipeline integrity', () => {
    it('matches extractor payloads to CaptureActivationSource getters across all captured stages', () => {
        expect(baseVectorLength).toBe(12);
        expect(tokenCount).toBeGreaterThan(0);
        expect(layers.length).toBeGreaterThan(0);
        expect(headCount).toBeGreaterThan(0);

        for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
            assertArrayClose(
                source.getEmbedding('token', tokenIndex, baseVectorLength),
                decodeVectorEntry(activations.embeddings.token[tokenIndex]),
                `embeddings.token token=${tokenIndex}`
            );
            assertArrayClose(
                source.getEmbedding('position', tokenIndex, baseVectorLength),
                decodeVectorEntry(activations.embeddings.position[tokenIndex]),
                `embeddings.position token=${tokenIndex}`
            );
            assertArrayClose(
                source.getEmbedding('sum', tokenIndex, baseVectorLength),
                decodeVectorEntry(activations.embeddings.sum[tokenIndex]),
                `embeddings.sum token=${tokenIndex}`
            );
        }

        for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
            const layer = layers[layerIndex];
            for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
                assertArrayClose(
                    source.getLayerIncoming(layerIndex, tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.incoming[tokenIndex]),
                    `layer=${layerIndex} incoming token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getLayerLn1(layerIndex, 'norm', tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.ln1.norm[tokenIndex]),
                    `layer=${layerIndex} ln1.norm token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getLayerLn1(layerIndex, 'scale', tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.ln1.scale[tokenIndex]),
                    `layer=${layerIndex} ln1.scale token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getLayerLn1(layerIndex, 'shift', tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.ln1.shift[tokenIndex]),
                    `layer=${layerIndex} ln1.shift token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getAttentionOutputProjection(layerIndex, tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.attn_output_proj[tokenIndex]),
                    `layer=${layerIndex} attention.output_projection token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getPostAttentionResidual(layerIndex, tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.post_attn_residual[tokenIndex]),
                    `layer=${layerIndex} residual.post_attention token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getLayerLn2(layerIndex, 'norm', tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.ln2.norm[tokenIndex]),
                    `layer=${layerIndex} ln2.norm token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getLayerLn2(layerIndex, 'scale', tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.ln2.scale[tokenIndex]),
                    `layer=${layerIndex} ln2.scale token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getLayerLn2(layerIndex, 'shift', tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.ln2.shift[tokenIndex]),
                    `layer=${layerIndex} ln2.shift token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getMlpUp(layerIndex, tokenIndex, 48),
                    decodeVectorEntry(layer.mlp_up[tokenIndex]),
                    `layer=${layerIndex} mlp.up token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getMlpActivation(layerIndex, tokenIndex, 48),
                    decodeVectorEntry(layer.mlp_act[tokenIndex]),
                    `layer=${layerIndex} mlp.activation token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getMlpDown(layerIndex, tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.mlp_down[tokenIndex]),
                    `layer=${layerIndex} mlp.down token=${tokenIndex}`
                );
                assertArrayClose(
                    source.getPostMlpResidual(layerIndex, tokenIndex, baseVectorLength),
                    decodeVectorEntry(layer.post_mlp_residual[tokenIndex]),
                    `layer=${layerIndex} residual.post_mlp token=${tokenIndex}`
                );
            }

            for (let headIndex = 0; headIndex < headCount; headIndex += 1) {
                for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
                    for (const kind of ['q', 'k', 'v']) {
                        const expected = decodeVectorEntry(layer.qkv[kind][headIndex][tokenIndex]);
                        const scalar = source.getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex);
                        assertNumberClose(
                            scalar,
                            expected[0],
                            `layer=${layerIndex} head=${headIndex} kind=${kind} token=${tokenIndex} scalar`
                        );
                        assertArrayClose(
                            source.getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, null),
                            expected,
                            `layer=${layerIndex} head=${headIndex} kind=${kind} token=${tokenIndex} vector`
                        );
                    }

                    const expectedPreRow = decodePackedAttentionRow(layer.attention_scores.pre[headIndex], tokenIndex);
                    const expectedPostRow = decodePackedAttentionRow(layer.attention_scores.post[headIndex], tokenIndex);
                    assertArrayClose(
                        source.getAttentionScoresRow(layerIndex, 'pre', headIndex, tokenIndex),
                        expectedPreRow,
                        `layer=${layerIndex} head=${headIndex} pre-row token=${tokenIndex}`
                    );
                    assertArrayClose(
                        source.getAttentionScoresRow(layerIndex, 'post', headIndex, tokenIndex),
                        expectedPostRow,
                        `layer=${layerIndex} head=${headIndex} post-row token=${tokenIndex}`
                    );

                    for (let keyTokenIndex = 0; keyTokenIndex < expectedPreRow.length; keyTokenIndex += 1) {
                        assertNumberClose(
                            source.getAttentionScore(layerIndex, 'pre', headIndex, tokenIndex, keyTokenIndex),
                            expectedPreRow[keyTokenIndex],
                            `layer=${layerIndex} head=${headIndex} pre-score q=${tokenIndex} k=${keyTokenIndex}`,
                            1e-4
                        );
                    }
                    for (let keyTokenIndex = 0; keyTokenIndex < expectedPostRow.length; keyTokenIndex += 1) {
                        assertNumberClose(
                            source.getAttentionScore(layerIndex, 'post', headIndex, tokenIndex, keyTokenIndex),
                            expectedPostRow[keyTokenIndex],
                            `layer=${layerIndex} head=${headIndex} post-score q=${tokenIndex} k=${keyTokenIndex}`,
                            1e-4
                        );
                    }

                    const weightedScalar = expectedPostRow.reduce((sum, weight, keyTokenIndex) => {
                        const valueScalar = decodeVectorEntry(layer.qkv.v[headIndex][keyTokenIndex])[0] ?? 0;
                        return sum + (weight * valueScalar);
                    }, 0);
                    assertArrayClose(
                        source.getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, null),
                        [weightedScalar],
                        `layer=${layerIndex} head=${headIndex} weighted-sum token=${tokenIndex}`
                    );
                }
            }
        }

        for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
            assertArrayClose(
                source.getFinalLayerNorm('norm', tokenIndex, baseVectorLength),
                decodeVectorEntry(activations.final_layernorm.norm[tokenIndex]),
                `final-layernorm.norm token=${tokenIndex}`
            );
            assertArrayClose(
                source.getFinalLayerNorm('scale', tokenIndex, baseVectorLength),
                decodeVectorEntry(activations.final_layernorm.scale[tokenIndex]),
                `final-layernorm.scale token=${tokenIndex}`
            );
            assertArrayClose(
                source.getFinalLayerNorm('shift', tokenIndex, baseVectorLength),
                decodeVectorEntry(activations.final_layernorm.shift[tokenIndex]),
                `final-layernorm.shift token=${tokenIndex}`
            );
        }
    });

    it('uses extracted values as the live color/debug inputs for residual, layernorm, mlp, and top final-layernorm vectors', () => {
        const layerIndex = Math.min(3, layers.length - 1);
        const tokenIndex = Math.min(2, tokenCount - 1);
        const tokenLabel = source.getTokenString(tokenIndex);
        const stageCases = [
            { stage: 'layer.incoming', values: source.getLayerIncoming(layerIndex, tokenIndex, baseVectorLength) },
            { stage: 'ln1.norm', values: source.getLayerLn1(layerIndex, 'norm', tokenIndex, baseVectorLength) },
            { stage: 'ln1.scale', values: source.getLayerLn1(layerIndex, 'scale', tokenIndex, baseVectorLength) },
            { stage: 'ln1.shift', values: source.getLayerLn1(layerIndex, 'shift', tokenIndex, baseVectorLength) },
            { stage: 'ln2.norm', values: source.getLayerLn2(layerIndex, 'norm', tokenIndex, baseVectorLength) },
            { stage: 'ln2.scale', values: source.getLayerLn2(layerIndex, 'scale', tokenIndex, baseVectorLength) },
            { stage: 'ln2.shift', values: source.getLayerLn2(layerIndex, 'shift', tokenIndex, baseVectorLength) },
            { stage: 'mlp.up_projection', values: source.getMlpUp(layerIndex, tokenIndex, 48) },
            { stage: 'mlp.down_projection', values: source.getMlpDown(layerIndex, tokenIndex, baseVectorLength) },
            { stage: 'residual.post_mlp', values: source.getPostMlpResidual(layerIndex, tokenIndex, baseVectorLength) },
        ];

        stageCases.forEach(({ stage, values }) => {
            const vector = createVectorDouble({ instanceCount: values.length });
            const applied = applyVectorData(
                vector,
                values,
                `Integrity ${stage}`,
                { stage, layerIndex, tokenIndex, tokenLabel }
            );

            expect(applied, stage).toBe(true);
            assertArrayClose(vector.rawData, values, `${stage} rawData`);
            expect(vector.updateKeyColorsFromData).toHaveBeenCalledTimes(1);
            assertArrayClose(
                vector.userData.activationData.values,
                values,
                `${stage} activationData.values`
            );
            expect(vector.userData.activationData.stage).toBe(stage);
            expect(vector.userData.activationData.layerIndex).toBe(layerIndex);
            expect(vector.userData.activationData.tokenIndex).toBe(tokenIndex);
        });

        const mlpActivation = source.getMlpActivation(layerIndex, tokenIndex, 48);
        const mlpSegments = sliceMlpActivationIntoSegments(mlpActivation, 4, baseVectorLength);
        expect(mlpSegments).toHaveLength(4);
        mlpSegments.forEach((segmentValues, segmentIndex) => {
            const vector = createVectorDouble({ instanceCount: segmentValues.length });
            applyVectorData(
                vector,
                segmentValues,
                `MLP Activation - ${tokenLabel}`,
                { stage: 'mlp.activation', layerIndex, tokenIndex, tokenLabel, segmentIndex }
            );
            assertArrayClose(vector.rawData, segmentValues, `mlp.activation segment=${segmentIndex} rawData`);
            assertArrayClose(
                vector.userData.activationData.values,
                segmentValues,
                `mlp.activation segment=${segmentIndex} activationData.values`
            );
            expect(vector.userData.activationData.segmentIndex).toBe(segmentIndex);
        });

        for (const stage of ['norm', 'scale', 'shift']) {
            const values = source.getFinalLayerNorm(stage, tokenIndex, baseVectorLength);
            const vector = createVectorDouble({ instanceCount: values.length });
            recolorVectorFromData(vector, values, null);
            assertArrayClose(vector.rawData, values, `final-layernorm.${stage} rawData`);
            expect(vector.updateKeyColorsFromData).toHaveBeenCalledTimes(1);
            expect(vector.userData.activationData).toBeUndefined();
        }
    });

    it('uses extracted values as the live color/debug inputs for qkv, weighted values, weighted sums, output projection, and post-attention residual', () => {
        const layerIndex = Math.min(2, layers.length - 1);
        const queryTokenIndex = Math.min(3, tokenCount - 1);
        const keyTokenIndex = Math.min(1, queryTokenIndex);
        const headIndex = Math.min(1, headCount - 1);
        const queryTokenLabel = source.getTokenString(queryTokenIndex);
        const keyTokenLabel = source.getTokenString(keyTokenIndex);

        for (const [kind, category] of [['q', 'Q'], ['k', 'K'], ['v', 'V']]) {
            const scalar = source.getLayerQKVScalar(layerIndex, kind, headIndex, queryTokenIndex);
            const vector = createVectorDouble({ instanceCount: 12, processed: true });
            vector.userData.parentLane = { tokenIndex: queryTokenIndex, tokenLabel: queryTokenLabel };
            vector.userData.headIndex = headIndex;

            const applied = applyQkvProcessedVisuals(
                vector,
                { activationSource: source, layerIndex },
                category,
                64,
                new THREE.Color(0xffffff)
            );

            expect(applied, `qkv.${kind}`).toBe(true);
            expect(vector.applyProcessedVisuals).toHaveBeenCalledTimes(1);
            assertArrayClose(vector.__processed.data, [scalar], `qkv.${kind} processedData`);
            expect(vector.__processed.numVisibleOutputUnits).toBe(64);
            assertArrayClose(vector.userData.activationData.values, [scalar], `qkv.${kind} activationData.values`);
            expect(vector.userData.activationData.stage).toBe(`qkv.${kind}`);
            expect(vector.userData.activationData.headIndex).toBe(headIndex);
            expect(vector.userData.activationData.tokenIndex).toBe(queryTokenIndex);
        }

        const postScore = source.getAttentionScore(layerIndex, 'post', headIndex, queryTokenIndex, keyTokenIndex);
        const valueScalar = source.getLayerQKVScalar(layerIndex, 'v', headIndex, keyTokenIndex);
        const weightedValue = [postScore * valueScalar];
        const weightedValueVector = createVectorDouble({ rawData: [valueScalar] });
        const weightedValueAnimator = {
            ctx: { layerIndex, outputVectorLength: 64 },
            _resolveOutputVectorLength: SelfAttentionAnimator.prototype._resolveOutputVectorLength,
            _sliceVectorData: SelfAttentionAnimator.prototype._sliceVectorData
        };
        SelfAttentionAnimator.prototype._tagConveyorValueVector.call(weightedValueAnimator, weightedValueVector, {
            lane: { tokenIndex: keyTokenIndex, tokenLabel: keyTokenLabel },
            queryLane: { tokenIndex: queryTokenIndex, tokenLabel: queryTokenLabel },
            headIdx: headIndex,
            weighted: true,
            valuesOverride: weightedValue,
            postScore
        });
        assertArrayClose(
            weightedValueVector.userData.activationData.values,
            weightedValue,
            'attention.weighted_value activationData.values'
        );
        expect(weightedValueVector.userData.activationData.stage).toBe('attention.weighted_value');
        expect(weightedValueVector.userData.activationData.queryTokenIndex).toBe(queryTokenIndex);
        expect(weightedValueVector.userData.activationData.keyTokenIndex).toBe(keyTokenIndex);
        assertNumberClose(
            weightedValueVector.userData.activationData.postScore,
            postScore,
            'attention.weighted_value postScore',
            1e-4
        );

        const weightedSumScalar = source.getAttentionWeightedSum(layerIndex, headIndex, queryTokenIndex, null)[0];
        const weightedSumVector = createVectorDouble({ instanceCount: 12, processed: true });
        weightedSumVector.userData.headIndex = headIndex;
        weightedSumVector.userData.parentLane = { tokenIndex: queryTokenIndex, tokenLabel: queryTokenLabel };
        const weightedSumAnimator = {
            ctx: { layerIndex, outputVectorLength: 64 },
            _valueHueRangeOptions: null,
            _resolveOutputVectorLength: SelfAttentionAnimator.prototype._resolveOutputVectorLength,
            _sliceVectorData: SelfAttentionAnimator.prototype._sliceVectorData,
            _syncWeightedSumActivationData: SelfAttentionAnimator.prototype._syncWeightedSumActivationData
        };
        SelfAttentionAnimator.prototype._applyWeightedSumScheme.call(weightedSumAnimator, weightedSumVector, [weightedSumScalar]);
        expect(weightedSumVector.applyProcessedVisuals).toHaveBeenCalledTimes(1);
        assertArrayClose(weightedSumVector.__processed.data, [weightedSumScalar], 'attention.weighted_sum processedData');
        expect(weightedSumVector.__processed.numVisibleOutputUnits).toBe(64);
        assertArrayClose(
            weightedSumVector.userData.activationData.values,
            [weightedSumScalar],
            'attention.weighted_sum activationData.values'
        );
        expect(weightedSumVector.userData.activationData.stage).toBe('attention.weighted_sum');

        const outputProjectionValues = source.getAttentionOutputProjection(layerIndex, queryTokenIndex, baseVectorLength);
        const outputProjectionVector = createVectorDouble({ instanceCount: baseVectorLength, processed: true });
        applyAttentionOutputProjectionDataToVector(outputProjectionVector, {
            activationSource: source,
            layerIndex,
            tokenIndex: queryTokenIndex,
            tokenLabel: queryTokenLabel,
            vectorPrismCount: baseVectorLength,
            outputVectorLength: 64
        });
        expect(outputProjectionVector.applyProcessedVisuals).toHaveBeenCalledTimes(1);
        assertArrayClose(outputProjectionVector.__processed.data, outputProjectionValues, 'attention.output_projection processedData');
        expect(outputProjectionVector.__processed.numVisibleOutputUnits).toBe(768);
        assertArrayClose(
            outputProjectionVector.userData.activationData.values,
            outputProjectionValues,
            'attention.output_projection activationData.values'
        );
        expect(outputProjectionVector.userData.activationData.stage).toBe('attention.output_projection');

        const postAttentionValues = source.getPostAttentionResidual(layerIndex, queryTokenIndex, baseVectorLength);
        const postAttentionVector = createVectorDouble({ instanceCount: baseVectorLength });
        const appliedPostResidual = applyPostAttentionResidualDataToVector(postAttentionVector, {
            values: postAttentionValues,
            layerIndex,
            tokenIndex: queryTokenIndex,
            tokenLabel: queryTokenLabel
        });
        expect(appliedPostResidual).toBe(true);
        assertArrayClose(postAttentionVector.rawData, postAttentionValues, 'residual.post_attention rawData');
        assertArrayClose(
            postAttentionVector.userData.activationData.values,
            postAttentionValues,
            'residual.post_attention activationData.values'
        );
        expect(postAttentionVector.userData.activationData.stage).toBe('residual.post_attention');
    });
});
