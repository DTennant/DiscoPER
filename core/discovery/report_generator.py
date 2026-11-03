# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Report generation for discovery runs."""

import json
import os
from typing import Any, Optional
from collections import Counter

from core.discovery.insight_store import InsightStore
from core.discovery.types import MetaInsight


class ReportGenerator:
    """Generates markdown reports from discovery runs with clear Level 1/Level 2 distinction."""

    def generate(
        self,
        insight_store: InsightStore,
        events_path: str,
        output_path: str,
        meta_insights: Optional[list[MetaInsight]] = None
    ):
        """
        Generate discovery report showing the flow from base insights to meta-insights.

        Args:
            insight_store: Store with all insights
            events_path: Path to events.jsonl
            output_path: Path to write report.md
            meta_insights: Optional list of meta-insights (observations)
        """
        # Load events
        events = []
        if os.path.exists(events_path):
            with open(events_path, "r") as f:
                for line in f:
                    events.append(json.loads(line))

        # Try to load meta-insights from JSON if not provided
        if meta_insights is None:
            workspace_root = os.path.dirname(events_path)
            meta_insights_path = os.path.join(workspace_root, "meta_insights.json")
            if os.path.exists(meta_insights_path):
                with open(meta_insights_path, "r") as f:
                    data = json.load(f)
                    meta_insights = [
                        MetaInsight.from_dict(mi) for mi in data.get("meta_insights", [])
                    ]

        # Separate base insights from meta-guided insights
        base_insights = [i for i in insight_store.insights if not getattr(i, 'is_meta', False)]
        meta_guided_insights = [i for i in insight_store.insights if getattr(i, 'is_meta', False)]

        # Count events by type
        event_counts = Counter(event["event_type"] for event in events)

        # Count insights by verdict
        base_verdict_counts = Counter(insight.verdict for insight in base_insights)
        meta_verdict_counts = Counter(insight.verdict for insight in meta_guided_insights)

        # Generate report
        report_lines = self._generate_header(
            base_insights, meta_guided_insights, meta_insights or [], event_counts
        )

        # Level 1: Base Discovery
        report_lines.extend(self._generate_level1_section(base_insights, base_verdict_counts))

        # Level 2: Meta-Discovery (if any)
        if meta_insights or meta_guided_insights:
            report_lines.extend(self._generate_level2_section(
                meta_insights or [], meta_guided_insights, meta_verdict_counts
            ))

        # Discovery Flow Visualization
        report_lines.extend(self._generate_flow_visualization(
            base_insights, meta_insights or [], meta_guided_insights
        ))

        # Timeline
        report_lines.extend(self._generate_timeline(events))

        # Write report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Report generated: {output_path}")

    def _generate_header(
        self,
        base_insights: list,
        meta_guided_insights: list,
        meta_insights: list,
        event_counts: Counter
    ) -> list[str]:
        """Generate report header with summary."""
        total_insights = len(base_insights) + len(meta_guided_insights)
        
        return [
            "# 🔬 Discovery System Report",
            "",
            "## Executive Summary",
            "",
            "```",
            "┌─────────────────────────────────────────────────────────────┐",
            "│                    DISCOVERY OVERVIEW                        │",
            "├─────────────────────────────────────────────────────────────┤",
            f"│  Level 1 (Base Discovery):     {len(base_insights):3d} insights                │",
            f"│  Level 2 (Meta-Discovery):     {len(meta_insights):3d} meta-insights           │",
            f"│  Level 2 (Guided Hypotheses):  {len(meta_guided_insights):3d} insights tested         │",
            "├─────────────────────────────────────────────────────────────┤",
            f"│  Total Insights:               {total_insights:3d}                          │",
            f"│  Total Hypotheses Tested:      {event_counts.get('hypothesis_proposed', 0):3d}                          │",
            "└─────────────────────────────────────────────────────────────┘",
            "```",
            "",
        ]

    def _generate_level1_section(self, base_insights: list, verdict_counts: Counter) -> list[str]:
        """Generate Level 1 (Base Discovery) section."""
        lines = [
            "---",
            "",
            "## 📊 Level 1: Base Discovery",
            "",
            "*Discoveries about raw data*",
            "",
            "### Summary",
            "",
            f"| Verdict | Count |",
            f"|---------|-------|",
            f"| ✅ Supported | {verdict_counts.get('supported', 0)} |",
            f"| ❌ Rejected | {verdict_counts.get('rejected', 0)} |",
            f"| ❓ Inconclusive | {verdict_counts.get('inconclusive', 0)} |",
            "",
            "### Base Insights",
            "",
        ]

        # Group by verdict
        supported = [i for i in base_insights if i.verdict == "supported"]
        rejected = [i for i in base_insights if i.verdict == "rejected"]
        inconclusive = [i for i in base_insights if i.verdict == "inconclusive"]

        if supported:
            lines.append("#### ✅ Supported Findings")
            lines.append("")
            for insight in supported:
                lines.extend([
                    f"**{insight.id[:8]}** (strength: {insight.strength:.3f})",
                    f"> {insight.claim}",
                    "",
                ])

        if rejected:
            lines.append("#### ❌ Rejected Hypotheses")
            lines.append("")
            for insight in rejected:
                lines.extend([
                    f"**{insight.id[:8]}** (strength: {insight.strength:.3f})",
                    f"> {insight.claim}",
                    "",
                ])

        if inconclusive:
            lines.append("#### ❓ Inconclusive Results")
            lines.append("")
            for insight in inconclusive:
                lines.extend([
                    f"**{insight.id[:8]}** (strength: {insight.strength:.3f})",
                    f"> {insight.claim}",
                    "",
                ])

        return lines

    def _generate_level2_section(
        self,
        meta_insights: list,
        meta_guided_insights: list,
        verdict_counts: Counter
    ) -> list[str]:
        """Generate Level 2 (Meta-Discovery) section."""
        lines = [
            "---",
            "",
            "## 🔍 Level 2: Meta-Discovery",
            "",
            "*Observations about patterns in base insights → Guided new hypotheses*",
            "",
        ]

        # Meta-Insights (observations, not tested)
        if meta_insights:
            lines.extend([
                "### Meta-Insights (Observations)",
                "",
                "*These are patterns observed across base insights. They guide future hypothesis generation but are NOT tested experimentally.*",
                "",
            ])

            for mi in meta_insights:
                emoji = self._get_meta_type_emoji(mi.meta_type)
                lines.extend([
                    f"#### {emoji} {mi.meta_type.replace('_', ' ').title()}",
                    "",
                    f"**Observation:** {mi.observation}",
                    "",
                    f"**→ Recommendation:** {mi.actionable_recommendation}",
                    "",
                    f"*Variables: {', '.join(mi.affected_variables)} | Priority: {mi.priority} | Confidence: {mi.confidence:.0%}*",
                    "",
                ])

        # Guided Insights (tested based on meta-insights)
        if meta_guided_insights:
            lines.extend([
                "### Guided Insights (Tested)",
                "",
                "*These hypotheses were designed based on meta-insight recommendations and then tested.*",
                "",
                "| Verdict | Count |",
                "|---------|-------|",
                f"| ✅ Supported | {verdict_counts.get('supported', 0)} |",
                f"| ❌ Rejected | {verdict_counts.get('rejected', 0)} |",
                f"| ❓ Inconclusive | {verdict_counts.get('inconclusive', 0)} |",
                "",
            ])

            for insight in meta_guided_insights:
                verdict_emoji = "✅" if insight.verdict == "supported" else "❌" if insight.verdict == "rejected" else "❓"
                lines.extend([
                    f"**{verdict_emoji} {insight.id[:8]}** (strength: {insight.strength:.3f})",
                    f"> {insight.claim}",
                    "",
                ])

        return lines

    def _get_meta_type_emoji(self, meta_type: str) -> str:
        """Get emoji for meta-insight type."""
        emoji_map = {
            "confound_pattern": "⚠️",
            "variable_cluster": "🔗",
            "success_pattern": "📈",
            "contradiction": "⚔️",
            "gap": "🕳️",
            "interaction_hint": "🔀",
        }
        return emoji_map.get(meta_type, "📌")

    def _generate_flow_visualization(
        self,
        base_insights: list,
        meta_insights: list,
        meta_guided_insights: list
    ) -> list[str]:
        """Generate ASCII visualization of the discovery flow."""
        lines = [
            "---",
            "",
            "## 🔄 Discovery Flow",
            "",
            "```",
        ]

        # Base insights summary
        supported_base = [i for i in base_insights if i.verdict == "supported"]
        lines.append("LEVEL 1: BASE DISCOVERY")
        lines.append("─" * 60)
        
        if supported_base:
            for i, insight in enumerate(supported_base[:5]):  # Limit to 5
                claim_short = insight.claim[:50] + "..." if len(insight.claim) > 50 else insight.claim
                lines.append(f"  • {claim_short}")
            if len(supported_base) > 5:
                lines.append(f"  ... and {len(supported_base) - 5} more supported findings")
        else:
            lines.append("  (no supported findings yet)")

        lines.append("")
        lines.append("          │")
        lines.append("          ▼")
        lines.append("")

        # Meta-insights
        lines.append("LEVEL 2: META-INSIGHTS (observations, NOT tested)")
        lines.append("─" * 60)
        
        if meta_insights:
            for mi in meta_insights[:3]:  # Limit to 3
                obs_short = mi.observation[:45] + "..." if len(mi.observation) > 45 else mi.observation
                rec_short = mi.actionable_recommendation[:40] + "..." if len(mi.actionable_recommendation) > 40 else mi.actionable_recommendation
                lines.append(f"  [{mi.meta_type}]")
                lines.append(f"    Observation: {obs_short}")
                lines.append(f"    → Action: {rec_short}")
                lines.append("")
            if len(meta_insights) > 3:
                lines.append(f"  ... and {len(meta_insights) - 3} more meta-insights")
        else:
            lines.append("  (no meta-insights generated)")

        lines.append("")
        lines.append("          │")
        lines.append("          ▼")
        lines.append("")

        # Guided hypotheses
        lines.append("LEVEL 2: GUIDED HYPOTHESES (tested)")
        lines.append("─" * 60)
        
        if meta_guided_insights:
            for insight in meta_guided_insights[:3]:  # Limit to 3
                verdict_icon = "✓" if insight.verdict == "supported" else "✗" if insight.verdict == "rejected" else "?"
                claim_short = insight.claim[:50] + "..." if len(insight.claim) > 50 else insight.claim
                lines.append(f"  [{verdict_icon}] {claim_short}")
            if len(meta_guided_insights) > 3:
                lines.append(f"  ... and {len(meta_guided_insights) - 3} more guided insights")
        else:
            lines.append("  (no guided hypotheses tested)")

        lines.append("```")
        lines.append("")

        return lines

    def _generate_timeline(self, events: list) -> list[str]:
        """Generate timeline section."""
        lines = [
            "---",
            "",
            "## 📅 Timeline",
            "",
        ]

        # Group events by iteration
        current_iteration = None
        for event in events:
            event_type = event.get("event_type", "unknown")
            is_meta = event.get("data", {}).get("is_meta", False)
            
            # Detect iteration changes from event data
            if "hypothesis_proposed" in event_type:
                prefix = "🔬 [META] " if is_meta else "🧪 "
                lines.append(f"{prefix}Hypothesis proposed")
            elif "experiment_planned" in event_type:
                prefix = "📋 [META] " if is_meta else "📋 "
                lines.append(f"{prefix}Experiment planned")
            elif "experiment_result" in event_type:
                lines.append(f"  📊 Result recorded")
            elif "insight_added" in event_type:
                prefix = "💡 [META] " if is_meta else "💡 "
                lines.append(f"{prefix}Insight added")
            elif "meta_discovery_step" in event_type:
                n_meta = event.get("data", {}).get("n_meta_insights_generated", 0)
                n_hyp = event.get("data", {}).get("n_guided_hypotheses_generated", 0)
                lines.append(f"🔍 **Meta-Discovery Step**: {n_meta} meta-insights → {n_hyp} guided hypotheses")

        lines.append("")

        return lines
