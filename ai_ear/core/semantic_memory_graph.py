#!/usr/bin/env python3
"""
Semantic Episodic Memory Graph for The AI Ear

A god-tier local memory system combining:
- Episodic memory: Specific acoustic events with full context and timestamp
- Semantic memory: Abstracted concepts extracted from episodes
- Graph structure: Rich relations (temporal, semantic, causal)
- Sparse recall & LLM-ready context generation

Designed for real-time acoustic intelligence with long-term coherence.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class MemoryNode:
    id: str
    type: str  # "episode" | "concept"
    content: Dict[str, Any]
    timestamp: str
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)


@dataclass
class MemoryEdge:
    source: str
    target: str
    relation: str  # temporal, semantic, causal, co_occurrence
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticEpisodicMemoryGraph:
    """God-tier semantic + episodic memory graph for acoustic intelligence."""

    def __init__(self, max_episodes: int = 500):
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []
        self.episode_index: Dict[str, str] = {}
        self.concept_index: Dict[str, str] = {}
        self.max_episodes = max_episodes

    def add_acoustic_event(self, analysis_result: Dict[str, Any], importance: float = 0.6) -> str:
        """Add a new acoustic episode from the analysis pipeline."""
        node_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        episode_node = MemoryNode(
            id=node_id,
            type="episode",
            content=analysis_result,
            timestamp=timestamp,
            importance=importance,
            tags=self._extract_tags(analysis_result)
        )
        self.nodes[node_id] = episode_node

        # Temporal linking
        if self.episode_index:
            prev_id = list(self.episode_index.values())[-1]
            self.edges.append(MemoryEdge(
                source=prev_id, target=node_id, relation="temporal", weight=1.0
            ))

        # Semantic concept extraction and linking
        concepts = self._extract_concepts(analysis_result)
        for concept in concepts:
            concept_id = self._get_or_create_concept(concept)
            self.edges.append(MemoryEdge(
                source=node_id, target=concept_id, relation="semantic", weight=0.8
            ))

        self.episode_index[timestamp] = node_id

        # Simple memory management
        if len(self.episode_index) > self.max_episodes:
            oldest = min(self.episode_index.keys())
            old_id = self.episode_index.pop(oldest)
            self._prune_node(old_id)

        return node_id

    def _extract_tags(self, result: Dict) -> List[str]:
        tags = []
        if result.get("speech"):
            tags.append("speech")
        if result.get("emotion"):
            tags.append(result["emotion"].get("dominant", "neutral"))
        if result.get("environment"):
            tags.append(result["environment"].get("scene", "unknown"))
        return tags

    def _extract_concepts(self, result: Dict) -> List[str]:
        concepts = []
        if result.get("speech") and result["speech"].get("text"):
            concepts.append("spoken_content")
        if result.get("emotion"):
            concepts.append(f"emotion_{result['emotion'].get('dominant', 'neutral')}")
        if result.get("environment"):
            concepts.append(f"env_{result['environment'].get('scene', 'unknown')}")
        return concepts

    def _get_or_create_concept(self, concept: str) -> str:
        if concept in self.concept_index:
            return self.concept_index[concept]
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = MemoryNode(
            id=node_id, type="concept",
            content={"name": concept},
            timestamp=datetime.now(timezone.utc).isoformat(),
            importance=0.4
        )
        self.concept_index[concept] = node_id
        return node_id

    def query_recent_context(self, max_nodes: int = 8) -> Dict[str, Any]:
        """Generate rich, LLM-ready context from recent episodes."""
        recent_nodes = list(self.nodes.values())[-max_nodes:]
        episodes = [n for n in recent_nodes if n.type == "episode"]

        key_concepts = list({n.content.get("name") for n in recent_nodes if n.type == "concept"})

        return {
            "recent_episodes": len(episodes),
            "key_concepts": key_concepts[:6],
            "summary": self._generate_summary(episodes),
            "events": [n.content for n in episodes[-3:]]
        }

    def _generate_summary(self, episodes: List[MemoryNode]) -> str:
        if not episodes:
            return "No recent acoustic activity detected."
        parts = []
        for ep in episodes[-3:]:
            c = ep.content
            if c.get("speech"):
                parts.append(f"Spoke: {c['speech'].get('text', '')[:70]}")
            if c.get("emotion"):
                parts.append(f"Emotion: {c['emotion'].get('dominant')}")
        return " | ".join(parts) if parts else "Quiet acoustic environment."

    def get_stats(self) -> Dict:
        return {
            "total_nodes": len(self.nodes),
            "episodes": len([n for n in self.nodes.values() if n.type == "episode"]),
            "concepts": len([n for n in self.nodes.values() if n.type == "concept"]),
            "edges": len(self.edges)
        }

    def _prune_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
