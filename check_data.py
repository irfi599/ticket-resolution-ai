from datasets import load_dataset
import re

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

dataset = load_dataset(
    "PRAPAREE/telecom-conversation-corpus",
    split="train"
)

# --------------------------------------------------
# SELECT ONE CONVERSATION
# --------------------------------------------------

conversation_id = "0e79164e7b8b4633bc1424637eacab32"

conversation = dataset.filter(
    lambda x: x["conversation_id"] == conversation_id
)

# --------------------------------------------------
# COLLECT CUSTOMER AND AGENT MESSAGES
# --------------------------------------------------

customer_messages = []
agent_messages = []

for row in conversation:

    if row["speaker"] == "client":
        customer_messages.append(row["text"])

    elif row["speaker"] == "agent":
        agent_messages.append(row["text"])


# --------------------------------------------------
# COMBINE CUSTOMER MESSAGES
# --------------------------------------------------

customer_text = " ".join(customer_messages)

# --------------------------------------------------
# EXTRACT DEVICE
# --------------------------------------------------

device_match = re.search(
    r"(iPhone\s+\d+(?:\s+Pro)?|Android|Samsung|Pixel)",
    customer_text,
    re.IGNORECASE
)

if device_match:
    device = device_match.group()
else:
    device = "Not identified"


# --------------------------------------------------
# DETECT ISSUE TYPE
# --------------------------------------------------

issue_types = []

if "dropped call" in customer_text.lower():
    issue_types.append("Dropped calls")

if "slow" in customer_text.lower() and "data" in customer_text.lower():
    issue_types.append("Slow mobile data")

if "poor reception" in customer_text.lower():
    issue_types.append("Poor reception")

if "network" in customer_text.lower():
    issue_types.append("Network issue")

if issue_types:
    issue_type = ", ".join(issue_types)
else:
    issue_type = "Other"


# --------------------------------------------------
# DETECT SENTIMENT
# --------------------------------------------------

negative_words = [
    "frustrating",
    "frustrated",
    "angry",
    "disappointed",
    "annoying",
    "terrible",
    "unhappy"
]

sentiment = "Neutral"

for word in negative_words:
    if word in customer_text.lower():
        sentiment = "Negative"
        break


# --------------------------------------------------
# EXTRACT TROUBLESHOOTING STEPS
# --------------------------------------------------

troubleshooting_steps = []

agent_text = " ".join(agent_messages).lower()

if "restart" in agent_text:
    troubleshooting_steps.append(
        "Restart the phone"
    )

if "software update" in agent_text:
    troubleshooting_steps.append(
        "Check for software updates"
    )

if "network mode" in agent_text:
    troubleshooting_steps.append(
        "Switch network mode"
    )

if "reset network settings" in agent_text:
    troubleshooting_steps.append(
        "Reset network settings"
    )


# --------------------------------------------------
# DETECT POSSIBLE OUTCOME
# --------------------------------------------------

if "engineer" in agent_text or "escalat" in agent_text:
    outcome = "Escalated"

elif troubleshooting_steps:
    outcome = "Troubleshooting provided"

else:
    outcome = "Unclear"


# --------------------------------------------------
# CREATE STRUCTURED CASE
# --------------------------------------------------

historical_case = {

    "case_id": conversation_id,

    "customer_problem": customer_messages[0],

    "device": device,

    "issue_type": issue_type,

    "sentiment": sentiment,

    "troubleshooting_steps": troubleshooting_steps,

    "outcome": outcome,

    "source_conversation": conversation_id
}


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print("\n========== AUTOMATICALLY EXTRACTED CASE ==========")

for key, value in historical_case.items():

    print(f"\n{key}: {value}")