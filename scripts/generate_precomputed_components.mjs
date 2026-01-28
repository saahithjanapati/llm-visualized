import fs from 'fs';
import * as THREE from 'three';
import { Buffer } from 'buffer';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';

// ---------------------------------------------------------------------------
// Minimal FileReader polyfill so GLTFExporter works in Node.js.
// ---------------------------------------------------------------------------

if (typeof globalThis.FileReader === 'undefined') {
    class FileReader {
        constructor() {
            this.onloadend = null; // callback
            this.result = null;
        }

        _finish(result) {
            this.result = result;
            if (typeof this.onloadend === 'function') {
                this.onloadend({ target: this });
            }
        }

        readAsArrayBuffer(blob) {
            blob.arrayBuffer().then(buf => this._finish(buf));
        }

        readAsDataURL(blob) {
            blob.arrayBuffer().then(buf => {
                const mime = blob.type || 'application/octet-stream';
                const base64 = Buffer.from(buf).toString('base64');
                this._finish(`data:${mime};base64,${base64}`);
            });
        }
    }

    globalThis.FileReader = FileReader;
}

import { LayerNormalizationVisualization } from '../src/components/LayerNormalizationVisualization.js';
import { WeightMatrixVisualization } from '../src/components/WeightMatrixVisualization.js';
import {
    LN_PARAMS,
    MHA_MATRIX_PARAMS,
    MLP_MATRIX_PARAMS_UP,
    MLP_MATRIX_PARAMS_DOWN
} from '../src/utils/constants.js';

import { MHA_OUTPUT_PROJECTION_MATRIX_PARAMS } from '../src/animations/LayerAnimationConstants.js';

// ---------------------------------------------------------------------------
// Helper functions mirroring the internal cache-key generators so that we can
// tag each exported mesh with a stable identifier.  These **must** stay in
// sync with the original component source files.
// ---------------------------------------------------------------------------
function wmKey(p) {
    return [p.width, p.height, p.depth, p.topWidthFactor, p.cornerRadius, p.numberOfSlits, p.slitWidth, p.slitDepthFactor, p.slitBottomWidthFactor, p.slitTopWidthFactor, 'high'].join('|');
}

function lnKey(p, segments = 64) {
    return [p.width, p.height, p.depth, p.wallThickness, p.numberOfHoles, p.holeWidth, p.holeWidthFactor, segments].join('|');
}

// ---------------------------------------------------------------------------
// Build the geometries exactly once using the existing visualization classes.
// Each heavy CSG operation executes here during the build step instead of in
// the browser.
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();

// --- LayerNorm geometry (shared by LN1 & LN2) ---
{
    const vis = new LayerNormalizationVisualization(new THREE.Vector3(), LN_PARAMS.width, LN_PARAMS.height, LN_PARAMS.depth, LN_PARAMS.wallThickness, LN_PARAMS.numberOfHoles, LN_PARAMS.holeWidth, LN_PARAMS.holeWidthFactor);
    const key = lnKey(LN_PARAMS);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) {
            obj.userData.cacheKey = `LN|${key}`;
        }
    });
    scene.add(vis.group);
}

// --- MHSA Q/K/V matrix geometry ---
{
    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(), MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth, MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits, MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor, MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor);
    const key = wmKey({ ...MHA_MATRIX_PARAMS });
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) {
            obj.userData.cacheKey = `WM|${key}`;
        }
    });
    scene.add(vis.group);
}

// --- MLP Up-projection matrix ---
{
    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 2000), MLP_MATRIX_PARAMS_UP.width, MLP_MATRIX_PARAMS_UP.height, MLP_MATRIX_PARAMS_UP.depth, MLP_MATRIX_PARAMS_UP.topWidthFactor, MLP_MATRIX_PARAMS_UP.cornerRadius, MLP_MATRIX_PARAMS_UP.numberOfSlits, MLP_MATRIX_PARAMS_UP.slitWidth, MLP_MATRIX_PARAMS_UP.slitDepthFactor, MLP_MATRIX_PARAMS_UP.slitBottomWidthFactor, MLP_MATRIX_PARAMS_UP.slitTopWidthFactor);
    const key = wmKey({ ...MLP_MATRIX_PARAMS_UP });
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) {
            obj.userData.cacheKey = `WM|${key}`;
        }
    });
    scene.add(vis.group);
}

// --- MLP Down-projection matrix ---
{
    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 4000), MLP_MATRIX_PARAMS_DOWN.width, MLP_MATRIX_PARAMS_DOWN.height, MLP_MATRIX_PARAMS_DOWN.depth, MLP_MATRIX_PARAMS_DOWN.topWidthFactor, MLP_MATRIX_PARAMS_DOWN.cornerRadius, MLP_MATRIX_PARAMS_DOWN.numberOfSlits, MLP_MATRIX_PARAMS_DOWN.slitWidth, MLP_MATRIX_PARAMS_DOWN.slitDepthFactor, MLP_MATRIX_PARAMS_DOWN.slitBottomWidthFactor, MLP_MATRIX_PARAMS_DOWN.slitTopWidthFactor);
    const key = wmKey({ ...MLP_MATRIX_PARAMS_DOWN });
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) {
            obj.userData.cacheKey = `WM|${key}`;
        }
    });
    scene.add(vis.group);
}

// --- MHSA Output Projection matrix ---
{
    const matrixHeight = MHA_MATRIX_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
    const params = {
        width: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
        height: matrixHeight,
        depth: MHA_MATRIX_PARAMS.depth, // use same depth as QKV matrices / lane dependent
        topWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
        cornerRadius: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
        numberOfSlits: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.numberOfSlits,
        slitWidth: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
        slitDepthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
        slitBottomWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
        slitTopWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
    };

    const vis = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 6000), params.width, params.height, params.depth, params.topWidthFactor, params.cornerRadius, params.numberOfSlits, params.slitWidth, params.slitDepthFactor, params.slitBottomWidthFactor, params.slitTopWidthFactor);

    const key = wmKey(params);
    vis.group.traverse(obj => {
        if (obj.isMesh && obj.geometry) {
            obj.userData.cacheKey = `WM|${key}`;
        }
    });
    scene.add(vis.group);
}
// ---------------------------------------------------------------------------
// Export to GLB (binary glTF)
// ---------------------------------------------------------------------------
const exporter = new GLTFExporter();

exporter.parse(
    scene,
    (result) => {
        let buffer;
        if (result instanceof ArrayBuffer) {
            buffer = Buffer.from(result);
        } else if (ArrayBuffer.isView(result)) { // Uint8Array etc.
            buffer = Buffer.from(result.buffer);
        } else {
            // Fallback – stringify JSON (should not happen when binary: true)
            buffer = Buffer.from(JSON.stringify(result), 'utf8');
        }
        fs.writeFileSync('precomputed_components.glb', buffer);
        console.log('✔  precomputed_components.glb generated');
    },
    (error) => {
        console.error('GLTF export failed:', error);
        process.exit(1);
    },
    { binary: true }
); 
