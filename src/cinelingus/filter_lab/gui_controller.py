from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from .gui_support import collect_parameter_values, render_parameter_controls
from .presentation import detail_text, relationship_summary
from .recipe import FilterRecipe, load_recipe, save_recipe
from .registry import default_filter_registry
from ..operator_language import display_mode_name, internal_mode_name


REGISTRY = default_filter_registry()


def current_filter_definition(app: Any):
    family_id = next((family.id for family in REGISTRY.families() if family.name == app.family_var.get()), "multiworld")
    return REGISTRY.get_in_family(family_id, app.mode_var.get())


def sync_filter_family(app: Any, preferred_filter: str | None = None) -> None:
    family_name = app.family_var.get()
    family_id = next((family.id for family in REGISTRY.families() if family.name == family_name), "translation")
    definitions = REGISTRY.filters_for_family(family_id)
    names = [display_mode_name(item.name) for item in definitions]
    app.mode_box.configure(values=names)
    implemented_default = next((display_mode_name(item.name) for item in definitions if item.implemented), names[0])
    app.mode_var.set(preferred_filter if preferred_filter in names else implemented_default)
    sync_filter_mode(app)


def sync_filter_mode(app: Any, *, existing_values: dict[str, Any] | None = None) -> None:
    definition = current_filter_definition(app)
    display_name = display_mode_name(definition.name)
    app._sync_film_selectors(definition)
    app.filter_detail_var.set(detail_text(definition))
    app.relationship_var.set(relationship_summary(definition))
    if definition.implemented:
        app.continue_button.configure(state="normal")
        maximum = "any number" if definition.maximum_films is None else str(definition.maximum_films)
        range_note = "" if definition.maximum_films == definition.minimum_films else f" (up to {maximum})"
        app.input_guidance_var.set(f"Choose {definition.minimum_films} film{'s' if definition.minimum_films != 1 else ''}{range_note}. Film A is the anchor.")
        app.status_var.set(f"Ready for {display_name}")
    else:
        app.continue_button.configure(state="disabled")
        app.input_guidance_var.set("This filter is not yet implemented.")
        app.status_var.set(f"{definition.name} - This filter is not yet implemented.")
    if hasattr(app, "filter_controls_frame"):
        app.filter_parameter_vars = render_parameter_controls(app.filter_controls_frame, definition, existing_values=existing_values)
    app._refresh_truth_panel()


def selected_filter_parameters(app: Any) -> dict[str, Any]:
    definition = current_filter_definition(app)
    if not definition.implemented:
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
    chosen = filedialog.asksaveasfilename(title="Save Filter Recipe", defaultextension=".json", filetypes=(("Filter recipe", "*.json"), ("All files", "*.*")))
    if chosen:
        save_recipe(recipe, Path(chosen))
        app.status_var.set(f"Saved recipe: {Path(chosen).name}")


def load_recipe_dialog(app: Any) -> None:
    chosen = filedialog.askopenfilename(title="Load Filter Recipe", filetypes=(("Filter recipe", "*.json"), ("All files", "*.*")))
    if not chosen:
        return
    try:
        loaded = load_recipe(Path(chosen))
        definition = REGISTRY.get(loaded.recipe.filter_id)
    except (ValueError, OSError) as exc:
        messagebox.showerror("Cannot load recipe", str(exc))
        return
    app.family_var.set(REGISTRY.family(definition.family_id).name)
    display_name = display_mode_name(definition.name)
    app.mode_var.set(display_name)
    sync_filter_family(app, preferred_filter=display_name)
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
