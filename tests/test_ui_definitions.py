from cinelingus.ui_definitions import setting_definition


def test_funniest_definition_discloses_mechanical_proxies() -> None:
    text = setting_definition("preference", "Funniest")
    assert "proxies" in text
    assert "contrast" in text


def test_every_visible_setting_group_has_a_fallback_definition() -> None:
    assert setting_definition("matching", "Unknown")


def test_translation_has_canonical_definition_and_transposition_migration_note() -> None:
    assert "complete supported timeline" in setting_definition("transformation", "Translation")
    assert "Legacy name" in setting_definition("transformation", "Transposition")
