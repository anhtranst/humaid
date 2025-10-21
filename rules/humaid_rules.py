# rules/humaid_rules.py

"""
HumAID zero-shot rules (project-local, experiment-focused).

- Put your evolving rule variants here: RULES_5 ... RULES_10
- Keep them short to control token costs.
- The label order matches README and humaidclf.prompts.LABELS.
"""

# Baseline template we can always fall back to
RULES_BASELINE = """
- caution_and_advice: Reports of warnings issued or lifted, guidance and tips related to the disaster.
- sympathy_and_support: Tweets with prayers, thoughts, and emotional support.
- requests_or_urgent_needs: Reports of urgent needs or supplies such as food, water, clothing, money,...
- displaced_people_and_evacuations: People who have relocated due to the crisis, even for a short time...
- injured_or_dead_people: Reports of injured or dead people due to the disaster.
- missing_or_found_people: Reports of missing or found people due to the disaster.
- infrastructure_and_utility_damage: Reports of any type of damage to infrastructure such as buildings, houses,...
- rescue_volunteering_or_donation_effort: Reports of any type of rescue, volunteering, or donation efforts...
- other_relevant_information: on-topic but none of the above
- not_humanitarian: If the tweet does not convey humanitarian aid-related information.
""".strip()

# === Experiment variants ===

RULES_1 = """
- requests_or_urgent_needs: asking for help/supplies/SOS
- rescue_volunteering_or_donation_effort: offering help, donation, organizing aid
- caution_and_advice: warnings/instructions/tips
- displaced_people_and_evacuations: evacuations, relocation, shelters
- injured_or_dead_people: injuries, casualties, fatalities
- missing_or_found_people: missing or found persons
- infrastructure_and_utility_damage: damage/outages to roads/bridges/power/water/buildings
- sympathy_and_support: prayers/condolences, no actionable info
- other_relevant_information: on-topic but none of the above
- not_humanitarian: unrelated to disasters/aid
Tie-break: prefer actionable class when in doubt.
""".strip()

RULES_2 = """
Pick ONE label for the tweet's PRIMARY INTENT.

- caution_and_advice: warnings/instructions/tips about the disaster
- sympathy_and_support: prayers/condolences/morale support (no logistics)
- requests_or_urgent_needs: asking for help/supplies/services (need/urgent/sos)
- displaced_people_and_evacuations: evacuation/relocation/shelter/displaced
- injured_or_dead_people: injuries/casualties/deaths
- missing_or_found_people: explicit missing OR found/reunited persons
- infrastructure_and_utility_damage: damage/outages to roads/buildings/power/water/comms caused by the disaster
- rescue_volunteering_or_donation_effort: offering help; organizing rescues/donations/volunteers/events
- other_relevant_information: on-topic facts/stats/official updates when none above fits
- not_humanitarian: unrelated to disasters or unclear context

Return only the label.
""".strip()

# Optional: a tiny registry so you can fetch by name
RULES_REGISTRY = {
    "BASELINE": RULES_BASELINE,
    "RULES_1": RULES_1,
    "RULES_2": RULES_2,
}

def get_rule(name: str) -> str:
    """Return the rule text by key in RULES_REGISTRY."""
    try:
        return RULES_REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown rule name: {name}. Available: {list(RULES_REGISTRY)}") from e
