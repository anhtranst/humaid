LABELS = [
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "requests_or_urgent_needs",
    "rescue_volunteering_or_donation_effort",
    "sympathy_and_support",
]

SYSTEM_PROMPT = (
  "You are a precise tweet classifier for humanitarian-response content. "
  "Choose exactly one label from the allowed list that best fits the tweet. "
  "If unrelated to humanitarian contexts, choose 'not_humanitarian'. "
  "Follow the short rules and output JSON that matches the schema; no extra fields."
)

def make_user_message(tweet_text: str, rules: str, labels: list[str] = LABELS) -> str:
    return (
        f"Allowed labels: {labels}\n"
        "Rules:\n"
        f"{rules.strip()}\n"
        "Choose exactly one label. If unrelated, choose 'not_humanitarian'.\n"
        f'Tweet: """{tweet_text}"""'
    )
