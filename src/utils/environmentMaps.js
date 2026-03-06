import * as THREE from 'three';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';

import roglandClearNightUrl from '../../rogland_clear_night_64.exr?url';
import autumnFieldUrl from '../../autumn_field_puresky_64.exr?url';
import qwantaniMoonNoonUrl from '../../qwantani_moon_noon_puresky_64.exr?url';

export const DEFAULT_ENVIRONMENT_KEY = 'rogland-clear-night';

export const ENVIRONMENT_MAP_OPTIONS = Object.freeze([
    Object.freeze({
        key: DEFAULT_ENVIRONMENT_KEY,
        label: 'Rogland Clear Night',
        url: roglandClearNightUrl
    }),
    Object.freeze({
        key: 'autumn-field-puresky',
        label: 'Autumn Field Pure Sky',
        url: autumnFieldUrl
    }),
    Object.freeze({
        key: 'qwantani-moon-noon-puresky',
        label: 'Qwantani Moon Noon Pure Sky',
        url: qwantaniMoonNoonUrl
    })
]);

const OPTION_MAP = new Map(ENVIRONMENT_MAP_OPTIONS.map((option) => [option.key, option]));
const loader = new EXRLoader().setDataType(THREE.HalfFloatType);
const textureCache = new Map();
const pendingLoads = new Map();

function configureEnvironmentTexture(texture) {
    if (!texture) return texture;
    texture.mapping = THREE.EquirectangularReflectionMapping;
    texture.center.set(0.5, 0.5);
    texture.rotation = Math.PI;
    texture.needsUpdate = true;
    return texture;
}

export function getEnvironmentMapOption(key) {
    return OPTION_MAP.get(key) || OPTION_MAP.get(DEFAULT_ENVIRONMENT_KEY) || ENVIRONMENT_MAP_OPTIONS[0];
}

export function loadEnvironmentMapTexture(key = DEFAULT_ENVIRONMENT_KEY) {
    const option = getEnvironmentMapOption(key);
    if (!option) {
        return Promise.reject(new Error(`Unknown environment map key: ${key}`));
    }
    if (textureCache.has(option.key)) {
        return Promise.resolve(textureCache.get(option.key));
    }
    if (pendingLoads.has(option.key)) {
        return pendingLoads.get(option.key);
    }

    const loadPromise = new Promise((resolve, reject) => {
        loader.load(
            option.url,
            (texture) => {
                const configured = configureEnvironmentTexture(texture);
                textureCache.set(option.key, configured);
                pendingLoads.delete(option.key);
                resolve(configured);
            },
            undefined,
            (error) => {
                pendingLoads.delete(option.key);
                reject(error);
            }
        );
    });

    pendingLoads.set(option.key, loadPromise);
    return loadPromise;
}
