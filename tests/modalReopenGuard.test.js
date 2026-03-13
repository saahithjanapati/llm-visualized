import { describe, expect, it } from 'vitest';

import { createModalReopenGuard } from '../src/ui/modalReopenGuard.js';

describe('modalReopenGuard', () => {
    it('blocks the first immediate reopen attempt after a close', () => {
        let currentTime = 100;
        const guard = createModalReopenGuard({ cooldownMs: 450, now: () => currentTime });

        expect(guard.shouldAllowOpen()).toBe(true);

        guard.markClosed();

        expect(guard.shouldAllowOpen()).toBe(false);
        expect(guard.shouldAllowOpen()).toBe(true);
    });

    it('allows reopen once the cooldown window has passed', () => {
        let currentTime = 100;
        const guard = createModalReopenGuard({ cooldownMs: 450, now: () => currentTime });

        guard.markClosed();
        currentTime = 600;

        expect(guard.shouldAllowOpen()).toBe(true);
        expect(guard.shouldAllowOpen()).toBe(true);
    });
});
