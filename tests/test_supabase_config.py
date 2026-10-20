from supabase_client import get_supabase_config


def test_supabase_config_has_no_hardcoded_fallback(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    assert get_supabase_config() == ("", "")