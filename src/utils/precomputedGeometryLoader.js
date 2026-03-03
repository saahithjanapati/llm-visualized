import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { USE_PRECOMPUTED_GEOMETRIES, USE_GLB_MATERIALS } from './constants.js';

function isPrecomputedGeometryDebugEnabled() {
    return typeof window !== 'undefined' && window.__PRECOMPUTED_GEOMETRY_DEBUG === true;
}

/**
 * Load a GLB containing pre-exported component geometries and inject them
 * into the runtime caches so that subsequent instantiations reuse the
 * already-baked BufferGeometry objects instead of running costly CSG builds.
 *
 * The exporter script encodes each mesh's cache-key in
 * `mesh.userData.cacheKey` and prefixes it with either "WM|" or "LN|" to
 * indicate which component the geometry belongs to (WeightMatrix or
 * LayerNorm).  We strip the prefix and forward the geometry to the relevant
 * component class.
 * 
 *
 * @param {string} [url='/precomputed_components.glb'] – Path or URL to the GLB asset.
 * @returns {Promise<void>} Resolves once the asset is fully loaded and caches are primed.
 */
export function loadPrecomputedGeometries(url = '../precomputed_components.glb') {
    const debugEnabled = isPrecomputedGeometryDebugEnabled();
    const globalToggle = (typeof window !== 'undefined' && typeof window.__USE_PRECOMPUTED_GEOMETRIES === 'boolean')
        ? window.__USE_PRECOMPUTED_GEOMETRIES
        : USE_PRECOMPUTED_GEOMETRIES;

    // URL query ?fresh=1 disables pre-baked geometries for this session
    const searchParams = (typeof window !== 'undefined') ? new URLSearchParams(window.location.search) : null;
    const queryDisables = searchParams ? searchParams.has('fresh') : false;

    if (!globalToggle || queryDisables) {
        if (debugEnabled) {
            console.log('[PrecomputedGeometryLoader] Skipping pre-baked geometry load (flag disabled)');
        }
        return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
        const timerLabel = '[PrecomputedGeometryLoader] Total load';
        let timed = false;
        if (debugEnabled) {
            console.time(timerLabel);
            timed = true;
            console.log(`[PrecomputedGeometryLoader] Fetching pre-baked geometries from ${url} …`);
        }

        const loader = new GLTFLoader();
        loader.load(
            url,
            (gltf) => {
                let wmCount = 0;
                let lnCount = 0;

                gltf.scene.traverse((obj) => {
                    if (obj.isMesh && obj.geometry && obj.userData && obj.userData.cacheKey) {
                        const cacheKey = obj.userData.cacheKey;
                        const material = USE_GLB_MATERIALS ? obj.material : null;
                        if (cacheKey.startsWith('WM|')) {
                            WeightMatrixVisualization.registerPrecomputedGeometry(cacheKey.substring(3), obj.geometry, material);
                            wmCount++;
                        } else if (cacheKey.startsWith('LN|')) {
                            LayerNormalizationVisualization.registerPrecomputedGeometry(cacheKey.substring(3), obj.geometry, material);
                            lnCount++;
                        }
                    }
                });

                if (timed) {
                    console.timeEnd(timerLabel);
                    console.log(`[PrecomputedGeometryLoader] Registered ${wmCount} WeightMatrix and ${lnCount} LayerNorm geometries.`);
                }
                resolve();
            },
            undefined,
            (err) => {
                console.warn('[PrecomputedGeometryLoader] Failed to load:', err);
                if (timed) {
                    console.timeEnd(timerLabel);
                }
                // Continue without cache so app still works.
                resolve();
            }
        );
    });
} 
