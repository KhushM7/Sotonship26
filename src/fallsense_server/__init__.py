"""fallsense server: turns backend outputs into incidents, carer alerts and the wall.

It consumes what the backend produces (skeletons, scores, thresholds and the
alert logic in `fallsense.calibrate`) and never re-implements it.
"""
