import * as THREE from 'three';

export default class BaseLayer {
    /**
     * @param {number} index - index of this layer in the transformer stack.
     */
    constructor(index = 0) {
        this.index = index;
        // Each concrete layer builds all of its meshes under this root so we
        // can move/clean the whole layer with a single group manipulation.
        this.root = new THREE.Group();
    }

    /**
     * Called once when the CoreEngine starts.  Adds the root group to the
     * shared scene so that subclasses can freely attach meshes to `this.root`.
     * @param {THREE.Scene} scene
     */
    init(scene) {
        scene.add(this.root);
    }

    /**
     * Called by CoreEngine every frame.  Subclasses override to implement
     * per-frame behaviour.  `dt` is the elapsed time in *seconds* already
     * multiplied by the current global speed factor so that layers do not
     * need to know about UI speed controls.
     *
     * @param {number} dt – delta-time in seconds
     */
    update(dt) {
        // intentionally empty – override in subclasses
    }

    /**
     * Dispose of geometries, materials and event handlers so we don't leak GPU
     * resources when the user navigates away.
     */
    dispose() {
        this.root.traverse(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                else obj.material.dispose();
            }
        });
    }
} 