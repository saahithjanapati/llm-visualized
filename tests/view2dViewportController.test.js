import { describe, expect, it } from 'vitest';
import {
    resolveViewportFitTransform,
    View2dViewportController
} from '../src/view2d/runtime/View2dViewportController.js';

describe('View2dViewportController', () => {
    it('fits bounds into the viewport with padding', () => {
        const transform = resolveViewportFitTransform(
            { x: 0, y: 0, width: 1600, height: 400 },
            { width: 800, height: 600 },
            { padding: 20, minScale: 0.1, maxScale: 4 }
        );

        expect(transform.scale).toBeCloseTo(0.475, 6);
        expect(transform.panX).toBeCloseTo(20, 6);
        expect(transform.panY).toBeCloseTo(205, 6);
    });

    it('fits the full scene and preserves anchor position during zoom', () => {
        const controller = new View2dViewportController({
            minScale: 0.1,
            maxScale: 4,
            padding: 20
        });
        controller.setViewportSize(800, 600);
        controller.setSceneBounds({ x: 0, y: 0, width: 1600, height: 400 });

        const fitted = controller.fitScene();
        expect(fitted.scale).toBeCloseTo(0.475, 6);

        const beforeWorld = controller.screenToWorld(400, 300);
        controller.zoomAt(1.5, 400, 300);
        const afterWorld = controller.screenToWorld(400, 300);

        expect(afterWorld.x).toBeCloseTo(beforeWorld.x, 6);
        expect(afterWorld.y).toBeCloseTo(beforeWorld.y, 6);
    });

    it('animates fly-to transitions toward target bounds', () => {
        const controller = new View2dViewportController({
            minScale: 0.1,
            maxScale: 4,
            padding: 40
        });
        controller.setViewportSize(800, 600);
        controller.setState({
            scale: 0.5,
            panX: 10,
            panY: 15
        });

        controller.flyToBounds(
            { x: 500, y: 100, width: 200, height: 100 },
            { animate: true, durationMs: 200, now: 1000, padding: 40 }
        );

        const mid = controller.step(1100);
        const end = controller.step(1200);

        expect(mid.scale).toBeGreaterThan(0.5);
        expect(end.scale).toBeCloseTo(3.6, 6);
        expect(end.panX).toBeCloseTo(-1760, 6);
        expect(end.panY).toBeCloseTo(-240, 6);
        expect(controller.getViewportTransform('test').source).toBe('test');
    });
});
