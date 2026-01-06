"""Influence maximization algorithms module."""

from .greedy import GreedyIM, LazyGreedyIM
from .tim import TIM, TIMPlus

__all__ = [
    'GreedyIM',
    'LazyGreedyIM',
    'TIM',
    'TIMPlus'
]
