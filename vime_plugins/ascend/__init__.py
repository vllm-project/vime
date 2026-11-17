"""Ascend NPU support for Vime.

Reached only through the seams on NPUAccelerator, which name the modules in
this package. Importing this package imports no vendor runtime: the Ascend
stack is loaded at the lifecycle points that own it.
"""
