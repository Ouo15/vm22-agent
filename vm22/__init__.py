"""Educational VM-22 (non-variable annuity PBR) valuation engine.

Everything in this package is deterministic and testable. It is a simplified
model for learning and demonstration: synthetic data, a stand-in economic
scenario generator and a synthetic mortality table. It is not a compliant
VM-22 valuation.
"""

DISCLAIMER = (
    "Educational model on synthetic data. Uses a stand-in interest-rate generator "
    "(not the NAIC GOES), a synthetic mortality table and simplified VM-22 mechanics. "
    "Not a compliant VM-22 valuation; results are illustrative only."
)
