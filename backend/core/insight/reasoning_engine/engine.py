from __future__ import annotations
from backend.knowledge_base.rules import RULES


class RuleEngine:
    def __init__(self, rules: list[dict] | None = None):
        src = rules if rules is not None else RULES
        self._rules = sorted(src, key=lambda r: r["priority"])

    # ── Public API ────────────────────────────────────────────────

    def suggest(self, domain: str, facts: dict) -> dict | None:
        """Return first (highest-priority) matching rule for domain."""
        for rule in self._rules:
            if rule["domain"] == domain and self._matches(rule["conditions"], facts):
                return self._format(rule, facts)
        return None

    def explain_all(self, domain: str, facts: dict) -> list[dict]:
        """Return all matching rules for domain (for educational display)."""
        return [
            self._format(rule, facts)
            for rule in self._rules
            if rule["domain"] == domain and self._matches(rule["conditions"], facts)
        ]

    def get_domain_rules(self, domain: str) -> list[dict]:
        """Return all rules in a domain (for display purposes)."""
        return [r for r in self._rules if r["domain"] == domain]

    # ── Internal ──────────────────────────────────────────────────

    def _matches(self, conditions: dict, facts: dict) -> bool:
        for key, cond in conditions.items():
            val = facts.get(key)

            # missing fact → condition fails (unless cond is "any")
            if val is None:
                return False

            if isinstance(cond, list):
                if val not in cond:
                    return False
            elif isinstance(cond, dict):
                if "min" in cond and val < cond["min"]:
                    return False
                if "max" in cond and val > cond["max"]:
                    return False
            elif isinstance(cond, bool):
                if bool(val) != cond:
                    return False
            else:
                if val != cond:
                    return False
        return True

    def _format(self, rule: dict, facts: dict) -> dict:
        return {
            "action":      rule["action"],
            "rule_id":     rule["id"],
            "domain":      rule["domain"],
            "priority":    rule["priority"],
            "explanation": rule["explanation"],
            "facts_used":  {k: facts.get(k) for k in rule["conditions"]},
        }


# Module-level singleton
_engine = RuleEngine()


def suggest(domain: str, facts: dict) -> dict | None:
    return _engine.suggest(domain, facts)


def explain_all(domain: str, facts: dict) -> list[dict]:
    return _engine.explain_all(domain, facts)


def get_domain_rules(domain: str) -> list[dict]:
    return _engine.get_domain_rules(domain)
