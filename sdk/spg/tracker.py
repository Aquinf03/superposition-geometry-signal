"""Tracker — live loss | geometry logger (superposition geometry beside loss)."""

from __future__ import annotations

from spg.signals import SignalLogger, StepRecord, format_live_line

# Public name for the SDK
Tracker = SignalLogger

__all__ = ["Tracker", "SignalLogger", "StepRecord", "format_live_line"]
