import json

import pytest

from bailiff.agents.groq_pool import GroqKeyPool


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_CONCURRENCY", raising=False)
    monkeypatch.delenv("DEFAULT_MAX_CONCURRENCY", raising=False)


def test_least_used_rotation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", json.dumps(["k1", "k2"]))
    pool = GroqKeyPool.from_env()

    with pool.acquire() as first:
        assert first.key == "k1"

    with pool.acquire() as second:
        assert second.key == "k2"


def test_concurrency_limits_force_failover(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", json.dumps(["k1", "k2"]))
    monkeypatch.setenv("GROQ_API_KEY_CONCURRENCY", json.dumps({"k1": 1, "k2": 2}))
    pool = GroqKeyPool.from_env()

    lease_one = pool.acquire()
    assert lease_one.key == "k1"

    lease_two = pool.acquire()
    assert lease_two.key == "k2"

    lease_two.mark_success()
    lease_one.mark_success()


def test_rate_limit_triggers_backoff(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", json.dumps(["k1", "k2"]))
    pool = GroqKeyPool.from_env()

    with pool.acquire() as lease:
        assert lease.key == "k1"
        lease.mark_rate_limited(RuntimeError("429"))

    with pool.acquire() as next_lease:
        assert next_lease.key == "k2"
        next_lease.mark_success()

