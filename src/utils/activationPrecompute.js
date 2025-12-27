import * as THREE from 'three';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { NUM_HEAD_SETS_LAYER, NUM_VECTOR_LANES } from './constants.js';

function getKeyColorCount(values) {
    const length = Array.isArray(values) ? values.length : 0;
    return Math.min(30, Math.max(1, length || 1));
}

function nextFrame() {
    if (typeof requestAnimationFrame === 'function') {
        return new Promise(resolve => requestAnimationFrame(resolve));
    }
    return new Promise(resolve => setTimeout(resolve, 0));
}

function primeVectorColors(vec, values, colorOptions = null) {
    if (!vec || !Array.isArray(values) || values.length === 0) return;
    vec.updateKeyColorsFromData(values, getKeyColorCount(values), colorOptions, values);
}

export async function precomputeActivationCaches(activationSource, options = {}) {
    if (!activationSource) return { ok: false };

    const laneTokenIndices = Array.isArray(options.laneTokenIndices)
        ? options.laneTokenIndices
        : activationSource.getLaneTokenIndices(NUM_VECTOR_LANES);
    const layerCount = Number.isFinite(options.layerCount)
        ? Math.max(1, Math.floor(options.layerCount))
        : (Array.isArray(activationSource.layers) ? activationSource.layers.length : 0);
    const baseLength = activationSource.getBaseVectorLength();
    const mlpLength = baseLength * 4;
    const yieldEveryLayer = Number.isFinite(options.yieldEveryLayer)
        ? Math.max(1, Math.floor(options.yieldEveryLayer))
        : 1;
    const report = typeof options.onProgress === 'function' ? options.onProgress : null;

    const scratchBase = new VectorVisualizationInstancedPrism(
        new Array(baseLength).fill(0),
        new THREE.Vector3(),
        30,
        baseLength
    );
    const scratchMlp = (mlpLength === baseLength)
        ? scratchBase
        : new VectorVisualizationInstancedPrism(
            new Array(mlpLength).fill(0),
            new THREE.Vector3(),
            30,
            mlpLength
        );

    let layersProcessed = 0;

    try {
        if (report) report({ phase: 'embeddings', message: 'Precomputing embeddings...' });
        laneTokenIndices.forEach((tokenIndex) => {
            primeVectorColors(
                scratchBase,
                activationSource.getEmbedding('token', tokenIndex, baseLength)
            );
            primeVectorColors(
                scratchBase,
                activationSource.getEmbedding('position', tokenIndex, baseLength)
            );
            primeVectorColors(
                scratchBase,
                activationSource.getEmbedding('sum', tokenIndex, baseLength)
            );
        });

        for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
            if (report) {
                report({
                    phase: 'layer',
                    layerIndex,
                    layerCount,
                    message: `Precomputing layer ${layerIndex + 1}/${layerCount}...`
                });
            }
            for (let t = 0; t < laneTokenIndices.length; t += 1) {
                const tokenIndex = laneTokenIndices[t];
                primeVectorColors(
                    scratchBase,
                    activationSource.getLayerIncoming(layerIndex, tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getLayerLn1(layerIndex, 'norm', tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getLayerLn1(layerIndex, 'scale', tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getLayerLn1(layerIndex, 'shift', tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getLayerLn2(layerIndex, 'norm', tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getLayerLn2(layerIndex, 'scale', tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getLayerLn2(layerIndex, 'shift', tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getAttentionOutputProjection(layerIndex, tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getPostAttentionResidual(layerIndex, tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getMlpDown(layerIndex, tokenIndex, baseLength)
                );
                primeVectorColors(
                    scratchBase,
                    activationSource.getPostMlpResidual(layerIndex, tokenIndex, baseLength)
                );

                primeVectorColors(
                    scratchMlp,
                    activationSource.getMlpUp(layerIndex, tokenIndex, mlpLength)
                );
                primeVectorColors(
                    scratchMlp,
                    activationSource.getMlpActivation(layerIndex, tokenIndex, mlpLength)
                );

                for (let headIndex = 0; headIndex < NUM_HEAD_SETS_LAYER; headIndex += 1) {
                    activationSource.getLayerQKVScalar(layerIndex, 'q', headIndex, tokenIndex);
                    activationSource.getLayerQKVScalar(layerIndex, 'k', headIndex, tokenIndex);
                    activationSource.getLayerQKVScalar(layerIndex, 'v', headIndex, tokenIndex);
                    activationSource.getAttentionScoresRow(layerIndex, 'pre', headIndex, tokenIndex);
                    activationSource.getAttentionScoresRow(layerIndex, 'post', headIndex, tokenIndex);
                }
            }

            layersProcessed += 1;
            if (yieldEveryLayer > 0 && layersProcessed % yieldEveryLayer === 0) {
                await nextFrame();
            }
        }
    } finally {
        scratchBase.dispose();
        if (scratchMlp !== scratchBase) {
            scratchMlp.dispose();
        }
    }

    if (report) report({ phase: 'done', message: 'Activation cache ready.' });
    return {
        ok: true,
        baseLength,
        mlpLength,
        layerCount,
        laneCount: laneTokenIndices.length
    };
}
