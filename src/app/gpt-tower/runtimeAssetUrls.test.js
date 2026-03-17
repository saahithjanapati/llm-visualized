import { describe, expect, it } from 'vitest';

import { resolveCaptureAssetUrl } from './runtimeAssetUrls.js';

describe('runtimeAssetUrls', () => {
    it('appends a generated version to known local capture files', () => {
        const resolvedUrl = new URL(resolveCaptureAssetUrl('capture.json'), 'https://example.test');

        expect(resolvedUrl.pathname).toBe('/capture.json');
        expect(resolvedUrl.searchParams.get('v')).toMatch(/^[a-f0-9]{10}$/);
    });

    it('preserves existing query params and hashes when versioning a known local capture', () => {
        const resolvedUrl = new URL(resolveCaptureAssetUrl('/capture.json?lane=4#focus'), 'https://example.test');

        expect(resolvedUrl.pathname).toBe('/capture.json');
        expect(resolvedUrl.searchParams.get('lane')).toBe('4');
        expect(resolvedUrl.searchParams.get('v')).toMatch(/^[a-f0-9]{10}$/);
        expect(resolvedUrl.hash).toBe('#focus');
    });

    it('does not overwrite an explicit version param on local capture URLs', () => {
        const resolvedUrl = new URL(resolveCaptureAssetUrl('/capture.json?v=manual&lane=4'), 'https://example.test');

        expect(resolvedUrl.pathname).toBe('/capture.json');
        expect(resolvedUrl.searchParams.get('v')).toBe('manual');
        expect(resolvedUrl.searchParams.get('lane')).toBe('4');
    });

    it('leaves external capture URLs untouched', () => {
        const externalUrl = 'https://cdn.example.com/capture.json?v=abc123';

        expect(resolveCaptureAssetUrl(externalUrl)).toBe(externalUrl);
    });

    it('passes through unknown local capture paths without adding a generated version', () => {
        expect(resolveCaptureAssetUrl('/capture_999.json?lane=4#focus')).toBe('/capture_999.json?lane=4#focus');
    });
});
