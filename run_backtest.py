"""Wrapper to run prosperity3bt with Round 1 product limits patched in."""
import prosperity3bt.runner as runner

# Patch in Round 1 position limits
runner.LIMITS['ASH_COATED_OSMIUM'] = 80
runner.LIMITS['INTARIAN_PEPPER_ROOT'] = 80

from prosperity3bt.__main__ import app
app()
