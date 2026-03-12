import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { NUM_HEAD_SETS_LAYER } from '../../utils/constants.js';
import { VectorRouter } from './VectorRouter.js';

function createVectorRef(position = new THREE.Vector3()) {
    const group = new THREE.Object3D();
    group.position.copy(position);
    group.visible = true;
    group.userData = {};
    return {
        rawData: [0],
        instanceCount: 1,
        group,
        userData: {},
    };
}

describe('VectorRouter', () => {
    it('marks travelling vectors dirty when an exact-arrival frame hides the final head copy', () => {
        const marked = [];
        const router = new VectorRouter(
            new THREE.Group(),
            Array.from({ length: NUM_HEAD_SETS_LAYER }, () => 0),
            Array.from({ length: NUM_HEAD_SETS_LAYER }, () => ({ q: 0, v: 0 })),
            0,
            [],
            {
                acquireVector: ({ position }) => createVectorRef(position),
                markVectorLayoutDirty: (vec) => {
                    marked.push(vec);
                    return true;
                }
            }
        );

        const travellingVec = createVectorRef(new THREE.Vector3(0, 0, 0));
        const lane = {
            horizPhase: 'travelMHSA',
            travellingVec,
            headIndex: NUM_HEAD_SETS_LAYER - 1,
            upwardCopies: [],
            sideCopies: []
        };

        router.update(1 / 60, 1000, [lane]);

        expect(travellingVec.group.visible).toBe(false);
        expect(lane.horizPhase).toBe('finishedHeads');
        expect(marked).toContain(travellingVec);
    });
});
