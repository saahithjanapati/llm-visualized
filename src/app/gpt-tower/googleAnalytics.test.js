// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
    initGoogleAnalyticsPageTracking,
    trackGoogleAnalyticsPageView
} from './googleAnalytics.js';

describe('googleAnalytics', () => {
    beforeEach(() => {
        window.history.replaceState({}, '', '/');
        window.gtag = vi.fn();
        document.title = 'LLM-Visualized';
        delete window.__llmVisualizedGoogleAnalyticsPageTracker;
        delete window.__llmVisualizedGoogleAnalyticsLastTrackedLocation;
    });

    afterEach(() => {
        if (window.__llmVisualizedGoogleAnalyticsPageTracker?.dispose) {
            window.__llmVisualizedGoogleAnalyticsPageTracker.dispose();
        }
        delete window.gtag;
        delete window.__llmVisualizedGoogleAnalyticsPageTracker;
        delete window.__llmVisualizedGoogleAnalyticsLastTrackedLocation;
    });

    it('sends a manual page_view with the full query string and route params', () => {
        window.history.replaceState({}, '', '/SA?token=4&generation=0&kvCache=0');

        const tracked = trackGoogleAnalyticsPageView(window.location, {
            referrer: 'https://example.com/landing'
        });

        expect(tracked).toBe(true);
        expect(window.gtag).toHaveBeenCalledTimes(1);
        expect(window.gtag).toHaveBeenCalledWith(
            'event',
            'page_view',
            expect.objectContaining({
                page_title: 'LLM-Visualized',
                page_location: 'http://localhost:3000/SA?token=4&generation=0&kvCache=0',
                page_referrer: 'https://example.com/landing',
                page_query_string: 'token=4&generation=0&kvCache=0',
                page_query_keys: 'token,generation,kvCache',
                route_token: '4',
                route_generation: '0',
                route_kv_cache: '0'
            })
        );
    });

    it('tracks history state URL changes and dedupes the follow-up initial send', () => {
        const tracker = initGoogleAnalyticsPageTracking({
            trackInitialPage: false
        });

        window.history.replaceState({}, '', '/?token=4&generation=0&kvCache=0');
        const trackedCurrent = tracker.trackCurrentPageView();

        expect(window.gtag).toHaveBeenCalledTimes(1);
        expect(trackedCurrent).toBe(false);
        expect(window.gtag).toHaveBeenLastCalledWith(
            'event',
            'page_view',
            expect.objectContaining({
                page_location: 'http://localhost:3000/?token=4&generation=0&kvCache=0',
                page_query_string: 'token=4&generation=0&kvCache=0'
            })
        );
    });

    it('uses the last tracked URL as the referrer for the next page_view', () => {
        const tracker = initGoogleAnalyticsPageTracking({
            trackInitialPage: true
        });

        expect(window.gtag).toHaveBeenCalledTimes(1);

        window.history.pushState({}, '', '/?token=5&generation=1&kvCache=1');

        expect(window.gtag).toHaveBeenCalledTimes(2);
        expect(window.gtag).toHaveBeenLastCalledWith(
            'event',
            'page_view',
            expect.objectContaining({
                page_location: 'http://localhost:3000/?token=5&generation=1&kvCache=1',
                page_referrer: 'http://localhost:3000/'
            })
        );

        tracker.dispose();
    });
});
