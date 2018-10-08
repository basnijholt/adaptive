# -*- coding: utf-8 -*-
import ipywidgets


def test_private_api_used_in_live_info():
    """We are catching all errors in
    adaptive.notebook_integration.should_update
    so if ipykernel changed its API it would happen unnoticed."""
    ipywidgets.Button().comm.kernel.iopub_thread._events
