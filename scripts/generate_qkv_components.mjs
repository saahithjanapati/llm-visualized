import fs from 'fs';
import * as THREE from 'three';
import { Buffer } from 'buffer';
import { dirname, resolve } from 'node:path';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const qkvOutputPath = resolve(scriptDir, '../src/assets/runtime/precomputed/precomputed_components_qkv.glb');

// ---------------------------------------------------------------------------
// Minimal FileReader polyfill so GLTFExporter works in Node.js.
// ---------------------------------------------------------------------------
if (typeof globalThis.FileReader === 'undefined') {
    class FileReader {
        constructor() {
            this.onloadend = null;
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

import { WeightMatrixVisualization } from '../src/components/WeightMatrixVisualization.js';
import { MHA_MATRIX_PARAMS } from '../src/utils/constants.js';
import { MHA_OUTPUT_PROJECTION_MATRIX_PARAMS } from '../src/animations/LayerAnimationConstants.js';

function wmKey(p) {
    return [p.width, p.height, p.depth, p.topWidthFactor, p.cornerRadius, p.numberOfSlits, p.slitWidth, p.slitDepthFactor, p.slitBottomWidthFactor, p.slitTopWidthFactor, 'high'].join('|');
}

const scene = new THREE.Scene();

// --- Q/K/V full-depth matrix geometry ---
{
    const params = { ...MHA_MATRIX_PARAMS };
    const vis = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(),
        params.width,
        params.height,
        params.depth,
        params.topWidthFactor,
        params.cornerRadius,
        params.numberOfSlits,
        params.slitWidth,
        params.slitDepthFactor,
        params.slitBottomWidthFactor,
        params.slitTopWidthFactor,
        false
    );
    const key = wmKey(params);
    if (vis.mesh && vis.mesh.geometry) {
        vis.mesh.userData.cacheKey = `WM|${key}`;
    }
    scene.add(vis.group);
}

// --- Output-projection full-depth matrix geometry ---
{
    const matrixHeight = MHA_MATRIX_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
    const params = {
        width: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
        height: matrixHeight,
        depth: MHA_MATRIX_PARAMS.depth,
        topWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
        cornerRadius: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
        numberOfSlits: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.numberOfSlits,
        slitWidth: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
        slitDepthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
        slitBottomWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
        slitTopWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
    };

    const vis = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(0, 0, 2000),
        params.width,
        params.height,
        params.depth,
        params.topWidthFactor,
        params.cornerRadius,
        params.numberOfSlits,
        params.slitWidth,
        params.slitDepthFactor,
        params.slitBottomWidthFactor,
        params.slitTopWidthFactor,
        false
    );

    const key = wmKey(params);
    if (vis.mesh && vis.mesh.geometry) {
        vis.mesh.userData.cacheKey = `WM|${key}`;
    }
    scene.add(vis.group);
}

const exporter = new GLTFExporter();

exporter.parse(
    scene,
    (result) => {
        let buffer;
        if (result instanceof ArrayBuffer) {
            buffer = Buffer.from(result);
        } else if (ArrayBuffer.isView(result)) {
            buffer = Buffer.from(result.buffer);
        } else {
            buffer = Buffer.from(JSON.stringify(result), 'utf8');
        }
        fs.mkdirSync(dirname(qkvOutputPath), { recursive: true });
        fs.writeFileSync(qkvOutputPath, buffer);
        console.log(`✔  ${qkvOutputPath} generated`);
    },
    (error) => {
        console.error('GLTF export failed:', error);
        process.exit(1);
    },
    { binary: true }
);
