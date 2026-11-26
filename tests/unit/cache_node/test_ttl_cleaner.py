"""Tests for the background expiry task.

These use asyncio.run() inside normal test functions, so they work with or
without the pytest-asyncio plugin installed.
"""
import asyncio

from app.cache.ttl_manager import TTLCleaner


def test_background_cleanup_removes_expired_keys_without_any_read(cache, clock):
    cache.set("dies", "v", ttl=1)
    cache.set("lives", "v")
    clock.advance(2)

    async def scenario():
        cleaner = TTLCleaner(cache, interval=0.01)
        cleaner.start()
        await asyncio.sleep(0.1)
        await cleaner.stop()

    asyncio.run(scenario())

    assert cache.size() == 1
    assert cache.stats.expired == 1
    assert cache.stats.gets == 0  # proves it was the cleaner, not a lazy GET


def test_cleaner_survives_a_failing_pass():
    class FlakyEngine:
        def __init__(self):
            self.calls = 0

        def purge_expired(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return 0

    engine = FlakyEngine()

    async def scenario():
        cleaner = TTLCleaner(engine, interval=0.01)
        cleaner.start()
        await asyncio.sleep(0.1)
        still_running = cleaner.is_running
        await cleaner.stop()
        return still_running

    assert asyncio.run(scenario()) is True
    assert engine.calls >= 2  # kept going after the first pass raised


def test_stop_is_safe_before_start_and_when_repeated(cache):
    async def scenario():
        cleaner = TTLCleaner(cache, interval=0.01)
        await cleaner.stop()  # never started
        cleaner.start()
        assert cleaner.is_running
        await cleaner.stop()
        await cleaner.stop()  # twice
        assert not cleaner.is_running

    asyncio.run(scenario())
