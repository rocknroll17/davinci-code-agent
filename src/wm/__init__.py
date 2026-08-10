"""World model (DreamerV3-style) for Da Vinci Code self-play.

Modules:
- nets:   RSSM (sequence model + categorical latents) and all predictor heads
- replay: episode-based sequence replay buffer
- dreamer: collection → world-model training → imagination actor-critic loop
"""
