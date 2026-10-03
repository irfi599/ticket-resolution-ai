from datasets import load_dataset
import re
import json

print("Loading telecom dataset...")

dataset = load_dataset(
    "PRAPAREE/telecom-conversation-corpus",
    split="train",
    streaming=True
)

print("Dataset streaming started.")

TARGET_CASES = 10000

conversations = {}

print(f"\nCollecting {TARGET_CASES} useful conversations...")

for row in dataset:

    conversation_id = row["conversation_id"]

    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "customer": [],
            "agent": []
        }

    cleaned_message = row["text"]

    if row["speaker"] == "client":
        conversations[conversation_id]["customer"].append(
            cleaned_message
        )

    elif row["speaker"] == "agent":
        conversations[conversation_id]["agent"].append(
            cleaned_message
        )

    if len(conversations) % 500 == 0:
        print(
            "Conversations collected:",
            len(conversations)
        )

    if len(conversations) >= TARGET_CASES:
        break


print("\nCollected conversations:", len(conversations))


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def clean_text(text):

    text = re.sub(
        r'\b(?:account\s+)?PIN\s*(?:is|:)?\s*\d{4,6}\b',
        '[REDACTED PIN]',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'\b\d{10}\b',
        '[REDACTED PHONE]',
        text
    )

    text = re.sub(
        r'\b(?:account\s*(?:number|no\.?)?)\s*[:#]?\s*\d{6,}\b',
        '[REDACTED ACCOUNT]',
        text,
        flags=re.IGNORECASE
    )

    return text


def extract_troubleshooting(agent_text):

    steps = []

    text = agent_text.lower()

    if "restart" in text:
        steps.append("Restart the phone")

    if "software update" in text or "software updates" in text:
        steps.append("Check for software updates")

    if "network mode" in text:
        steps.append("Switch network mode")

    if "reset network settings" in text:
        steps.append("Reset network settings")

    if "engineering" in text or "engineer" in text:
        steps.append("Escalate the issue to engineering")

    if "dedicated support" in text:
        steps.append("Refer the customer to dedicated support")

    return steps


def determine_outcome(agent_text, troubleshooting_steps):

    text = agent_text.lower()

    if "engineer" in text or "engineering" in text:
        return "Escalated"

    if "dedicated support" in text:
        return "Referred to dedicated support"

    if troubleshooting_steps:
        return "Troubleshooting provided"

    return "Unclear"


# --------------------------------------------------
# Create historical cases
# --------------------------------------------------

print("\nCreating historical cases...")

historical_cases = []

for conversation_id, messages in conversations.items():

    customer_messages = [
        clean_text(x)
        for x in messages["customer"]
    ]

    agent_messages = [
        clean_text(x)
        for x in messages["agent"]
    ]

    customer_text = " ".join(customer_messages)
    agent_text = " ".join(agent_messages)

    if not customer_text:
        continue

    customer_lower = customer_text.lower()

    issue_types = []

    if (
        "dropped call" in customer_lower
        or "dropped calls" in customer_lower
    ):
        issue_types.append("Dropped calls")

    if (
        "slow" in customer_lower
        and (
            "data" in customer_lower
            or "internet" in customer_lower
        )
    ):
        issue_types.append("Slow mobile data")

    if (
        "poor reception" in customer_lower
        or "weak signal" in customer_lower
    ):
        issue_types.append("Poor reception")

    if "network" in customer_lower:
        issue_types.append("Network issue")

    issue_type = (
        ", ".join(issue_types)
        if issue_types
        else "Other"
    )

    negative_words = [
        "frustrating",
        "frustrated",
        "angry",
        "disappointed",
        "annoying",
        "terrible",
        "unhappy",
        "hate",
        "worst"
    ]

    sentiment = "Neutral"

    for word in negative_words:
        if word in customer_lower:
            sentiment = "Negative"
            break

    device_match = re.search(
        r"(iPhone\s+\d+(?:\s+Pro)?|Android|Samsung|Pixel)",
        customer_text,
        re.IGNORECASE
    )

    device = (
        device_match.group()
        if device_match
        else "Not identified"
    )

    troubleshooting_steps = extract_troubleshooting(
        agent_text
    )

    outcome = determine_outcome(
        agent_text,
        troubleshooting_steps
    )

    historical_cases.append({

        "case_id": conversation_id,

        "customer_problem": customer_messages[0],

        "problem_text": customer_text,

        "customer_text": customer_text,

        "device": device,

        "issue_type": issue_type,

        "sentiment": sentiment,

        "troubleshooting_steps": troubleshooting_steps,

        "outcome": outcome,

        "source_conversation": conversation_id
    })


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(
    "historical_cases_10000.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        historical_cases,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n==========================================")
print("10,000-case preparation complete!")
print("==========================================")

print(
    "Historical cases created:",
    len(historical_cases)
)

print(
    "Saved as: historical_cases_10000.json"
)

print("==========================================")