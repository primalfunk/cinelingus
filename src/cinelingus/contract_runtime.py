from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

from .contract_kernel import RunContract

if TYPE_CHECKING:
    from .multi_input_guarantee import MultiInputGuarantee
    from .qualification import ScheduleQualification


_ACTIVE_CONTRACT: ContextVar[RunContract | None] = ContextVar("cinelingus_active_run_contract", default=None)
_ACTIVE_QUALIFICATION: ContextVar[ScheduleQualification | None] = ContextVar("cinelingus_active_schedule_qualification", default=None)
_ACTIVE_MULTI_INPUT_GUARANTEE: ContextVar[MultiInputGuarantee | None] = ContextVar("cinelingus_active_multi_input_guarantee", default=None)


def active_run_contract() -> RunContract | None:
    return _ACTIVE_CONTRACT.get()


def active_schedule_qualification() -> ScheduleQualification | None:
    return _ACTIVE_QUALIFICATION.get()


def active_multi_input_guarantee() -> MultiInputGuarantee | None:
    return _ACTIVE_MULTI_INPUT_GUARANTEE.get()


def record_schedule_qualification(qualification: ScheduleQualification) -> None:
    _ACTIVE_QUALIFICATION.set(qualification)


def record_multi_input_guarantee(guarantee: MultiInputGuarantee) -> None:
    _ACTIVE_MULTI_INPUT_GUARANTEE.set(guarantee)


@contextmanager
def activate_run_contract(contract: RunContract) -> Iterator[RunContract]:
    contract_token = _ACTIVE_CONTRACT.set(contract)
    qualification_token = _ACTIVE_QUALIFICATION.set(None)
    guarantee_token = _ACTIVE_MULTI_INPUT_GUARANTEE.set(None)
    try:
        yield contract
    finally:
        _ACTIVE_MULTI_INPUT_GUARANTEE.reset(guarantee_token)
        _ACTIVE_QUALIFICATION.reset(qualification_token)
        _ACTIVE_CONTRACT.reset(contract_token)
