"""
CADMUS — Circuit-Aware Dynamic Multilayer Update System
Adaptive full-stack quantum error correction for NISQ hardware.
"""

__version__ = "0.1.0"
__author__ = "Nicolas Corredor Guasca"
__license__ = "Apache-2.0"

from cadmus.core import CADMUS
from cadmus.profiler.health_check import CircuitHealthCheck
from cadmus.profiler.noise_profiler import NoiseProfiler
from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
from cadmus.correction.readout import ReadoutCorrector
from cadmus.correction.mid_circuit import MidCircuitCorrector
from cadmus.decoder.hybrid_decoder import HybridDecoder
from cadmus.drift.detector import SyndromeDriftDetector

__all__ = [
    "CADMUS",
    "CircuitHealthCheck",
    "NoiseProfiler",
    "AdaptiveEncoder",
    "ReadoutCorrector",
    "MidCircuitCorrector",
    "HybridDecoder",
    "SyndromeDriftDetector",
]
