"""
Research scaffolding — implemented and unit-tested, NOT validated at scale.

Nothing in here backs a claim anywhere in this repository. It lives in its own
subpackage so that "AtlasInfer's public API" and "things that are half-explored"
are visibly different sets, and so a reader evaluating the library is not left
guessing which is which.

Modules:
  * ``codebook`` — sub-4-bit vector quantization (2-3 bit). The frontier this
    reaches for (AQLM, QuIP#, QTIP) is established on 7B-13B models with the full
    WikiText-2/C4 protocol; that does not fit the hardware this was developed on,
    so no competitive claim is made. See the module docstring.
"""
