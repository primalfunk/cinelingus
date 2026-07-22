from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from .gui_support import collect_parameter_values, render_parameter_controls
from .presentation import detail_text, relationship_summary
from .public_catalog import OperatingMode, default_public_apparatus_catalog
from .recipe import FilterRecipe, load_recipe, save_recipe
from .registry import default_filter_registry
from ..operator_language import display_mode_name, internal_mode_name


REGISTRY = default_filter_registry()
CATALOG = default_public_apparatus_catalog()
REALITY_LABELS = {"One Film": OperatingMode.SOLITARY, "Several Films": OperatingMode.MULTIWORLD}


def _operating_mode(app: Any) -> OperatingMode:
    value = app.operating_mode_var.get()
    if value in REALITY_LABELS:
        return REALITY_LABELS[value]
    return OperatingMode(value)


def current_apparatus_entry(app: Any):
    return CATALOG.resolve(
        app.mode_var.get(),
        operating_mode=_operating_mode(app),
        discipline=app.family_var.get(),
    )


def current_filter_definition(app: Any):
    # Retain the old family-based lookup for small integrations that have not
    # yet adopted the public operating-mode variable. The production GUI does.
    if not hasattr(app, "operating_mode_var"):
        family_id = next((family.id for family in REGISTRY.families() if family.name == app.family_var.get()), "multiworld")
        return REGISTRY.get_in_family(family_id, app.mode_var.get())
    return REGISTRY.get(current_apparatus_entry(app).internal_id)


def sync_operating_mode(app: Any, preferred_discipline: str | None = None, preferred_apparatus: str | None = None) -> None:
    mode = _operating_mode(app)
    available_entries = CATALOG.entries(operating_mode=mode, primary_only=True)
    discipline_ids = {entry.discipline for entry in available_entries}
    discipline_names = [item.name for item in CATALOG.disciplines() if item.id in discipline_ids]
    if hasattr(app, "family_box"):
        app.family_box.configure(values=discipline_names)
    selected = preferred_discipline if preferred_discipline in discipline_names else app.family_var.get()
    if selected not in discipline_names:
        selected = discipline_names[0]
    app.family_var.set(selected)
    sync_filter_family(app, preferred_filter=preferred_apparatus)


def sync_filter_family(app: Any, preferred_filter: str | None = None) -> None:
    if not hasattr(app, "operating_mode_var"):
        family_name = app.family_var.get()
        family_id = next((family.id for family in REGISTRY.families() if family.name == family_name), "translation")
        definitions = REGISTRY.filters_for_family(family_id)
        names = [display_mode_name(item.name) for item in definitions]
        app.mode_box.configure(values=names)
        implemented_default = next((display_mode_name(item.name) for item in definitions if item.implemented), names[0])
        app.mode_var.set(preferred_filter if preferred_filter in names else implemented_default)
        sync_filter_mode(app)
        return
    entries = CATALOG.entries(
        operating_mode=_operating_mode(app),
        discipline=app.family_var.get(),
        primary_only=True,
    )
    names = [entry.public_name for entry in entries]
    app.mode_box.configure(values=names)
    preferred_name = None
    if preferred_filter:
        try:
            preferred_name = CATALOG.resolve(
                preferred_filter,
                operating_mode=_operating_mode(app),
                discipline=app.family_var.get(),
            ).public_name
        except ValueError:
            preferred_name = None
    app.mode_var.set(preferred_name if preferred_name in names else names[0])
    sync_filter_mode(app)


def _apparatus_detail(entry: Any, definition: Any) -> str:
    discipline = CATALOG.discipline(entry.discipline)
    reality = "ONE FILM" if entry.operating_mode == OperatingMode.SOLITARY else "SEVERAL FILMS"
    maximum = "unlimited" if entry.maximum_films is None else str(entry.maximum_films)
    unavailable = f"\n{entry.unavailable_reason}" if entry.unavailable_reason else ""
    limitations = " ".join(definition.known_limitations)
    caveat = f"\nKnown limitation: {limitations}" if limitations else ""
    return (
        f"{discipline.name} / {reality} / {entry.status.value.upper()}\n"
        f"{entry.public_description}\n"
        f"Law: {entry.short_law} Requires {entry.minimum_films}-{maximum} films; "
        f"capability tier {entry.minimum_capability_tier.value}.{unavailable}{caveat}"
    )


def sync_filter_mode(app: Any, *, existing_values: dict[str, Any] | None = None) -> None:
    definition = current_filter_definition(app)
    entry = CATALOG.get(definition.id) if hasattr(app, "operating_mode_var") else None
    display_name = entry.public_name if entry else display_mode_name(definition.name)
    app._sync_film_selectors(definition)
    if entry:
        app.filter_detail_var.set(_apparatus_detail(entry, definition))
    else:
        app.filter_detail_var.set(detail_text(definition))
    app.relationship_var.set(relationship_summary(definition))
    if definition.implemented and (entry is None or entry.invokable):
        app.continue_button.configure(state="normal")
        maximum = "any number" if definition.maximum_films is None else str(definition.maximum_films)
        range_note = "" if definition.maximum_films == definition.minimum_films else f" (up to {maximum})"
        app.input_guidance_var.set(f"Choose {definition.minimum_films} film{'s' if definition.minimum_films != 1 else ''}{range_note}. Film A is the anchor.")
        app.status_var.set(f"Ready for {display_name}")
    else:
        app.continue_button.configure(state="disabled")
        app.input_guidance_var.set(entry.unavailable_reason if entry else "This apparatus is dormant.")
        app.status_var.set(f"{display_name} — DORMANT")
    if hasattr(app, "filter_controls_frame"):
        app.filter_parameter_vars = render_parameter_controls(app.filter_controls_frame, definition, existing_values=existing_values)
    app._refresh_truth_panel()


def selected_filter_parameters(app: Any) -> dict[str, Any]:
    definition = current_filter_definition(app)
    if hasattr(app, "operating_mode_var"):
        current_apparatus_entry(app).require_invokable()
    elif not definition.implemented:
        raise ValueError(f"{definition.name} is in development and cannot be run.")
    return collect_parameter_values(definition, app.filter_parameter_vars)


def save_recipe_dialog(app: Any) -> None:
    try:
        definition = current_filter_definition(app)
        config = app._selected_config()
        parameters = selected_filter_parameters(app)
        roles = {"films": [str(path) for path in config.films]} if "films" in definition.required_inputs else {
            role: str(config.destination_video if role in {"film", "destination_video"} else config.source_dialogue)
            for role in definition.required_inputs
        }
        recipe = FilterRecipe.create(
            definition.id,
            input_media_roles=roles,
            parameters=parameters,
            output_settings={"form": "full_length", "output_directory": str(config.output_dir)},
            random_seed=int(parameters.get("seed", 1)),
            target_duration=None,
            requested_analysis_backends={"diarization": config.speaker_diarization_backend, "transcription": config.whisper_model},
        )
    except (ValueError, OSError) as exc:
        messagebox.showerror("Cannot save recipe", str(exc))
        return
    chosen = filedialog.asksaveasfilename(title="Save Apparatus Recipe", defaultextension=".json", filetypes=(("Apparatus recipe", "*.json"), ("All files", "*.*")))
    if chosen:
        save_recipe(recipe, Path(chosen))
        app.status_var.set(f"Saved recipe: {Path(chosen).name}")


def load_recipe_dialog(app: Any) -> None:
    chosen = filedialog.askopenfilename(title="Load Apparatus Recipe", filetypes=(("Apparatus recipe", "*.json"), ("All files", "*.*")))
    if not chosen:
        return
    try:
        loaded = load_recipe(Path(chosen))
        definition = REGISTRY.get(loaded.recipe.filter_id)
    except (ValueError, OSError) as exc:
        messagebox.showerror("Cannot load recipe", str(exc))
        return
    entry = CATALOG.get(definition.id)
    app.operating_mode_var.set("One Film" if entry.operating_mode == OperatingMode.SOLITARY else "Several Films")
    app.family_var.set(CATALOG.discipline(entry.discipline).name)
    display_name = entry.public_name
    app.mode_var.set(display_name)
    sync_operating_mode(app, preferred_discipline=app.family_var.get(), preferred_apparatus=display_name)
    sync_filter_mode(app, existing_values=loaded.recipe.parameters)
    roles = loaded.recipe.input_media_roles
    if "films" in roles:
        app._set_film_paths([Path(path) for path in roles["films"]])
    elif "film" in roles:
        app.destination_var.set(roles["film"])
        app.destination_selected_by_user = True
    else:
        if roles.get("destination_video"):
            app.destination_var.set(roles["destination_video"])
            app.destination_selected_by_user = True
        if roles.get("source_dialogue"):
            app.source_var.set(roles["source_dialogue"])
            app.source_selected_by_user = True
    output_dir = loaded.recipe.output_settings.get("output_directory")
    if output_dir:
        app.output_var.set(str(output_dir))
    app._show_wizard_step(2)
    notes = [*loaded.migrations, *loaded.warnings]
    suffix = '; '.join(notes)
    app.status_var.set(f'Loaded {display_name} recipe' + (f' - {suffix}' if suffix else ''))
