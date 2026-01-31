# AD Test Requirements

## Goal
Implement computational graph that can:
1. Load from JSON description
2. Execute with same numerical output as example/ymodel2-s-2

## Current Status
Starting implementation - this requires:
- Full operator execution (Linear, RMSNorm, Attention, FFN, etc.)
- Weight loading
- Exact numerical precision matching
- Same random seed handling

This is a substantial task requiring complete inference engine via computational graph.
