from datasets import load_dataset
import re
import json

# ==========================================
# LOAD DATASET
# ==========================================

dataset = load_dataset(
    "PRAPAREE/telecom-conversation-corpus",
    split="train"
)

print("Dataset loaded successfully!")


# ==========================================
# CONVERSATIONS WE WANT TO USE
# ==========================================

conversation_ids = [
    "0e79164e7b8b4633bc1424637eacab32",
    "98a6dbc997614ceb8e0734dda38b2524",
    "9db3dbf522c3487691ce42dd23658e0a"
]


# ==========================================
# CLEAN SENSITIVE INFORMATION
# ==========================================

def clean_text(text):

    # Redact PIN
    text = re.sub(
        r'\b(?:account\s+)?PIN\s*(?:is|:)?\s*\d{4,6}\b',
        '[REDACTED PIN]',
        text,
        flags=re.IGNORECASE
    )

    # Redact phone numbers
    text = re.sub(
        r'\b\d{10}\b',
        '[REDACTED PHONE]',
        text
    )

    # Redact account numbers
    text = re.sub(
        r'\b(?:account\s*(?:number|no\.?)?)\s*[:#]?\s*\d{6,}\b',
        '[REDACTED ACCOUNT]',
        text,
        flags=re.IGNORECASE
    )

    return text


# ==========================================
# EXTRACT TROUBLESHOOTING STEPS
# ==========================================

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


# ==========================================
# DETERMINE OUTCOME
# ==========================================

def determine_outcome(agent_text, troubleshooting_steps):

    text = agent_text.lower()

    if "engineer" in text or "engineering" in text:
        return "Escalated"

    if "dedicated support" in text:
        return "Referred to dedicated support"

    if troubleshooting_steps:
        return "Troubleshooting provided"

    return "Unclear"


# ==========================================
# EXTRACT HISTORICAL CASES
# ==========================================

historical_cases = []


for conversation_id in conversation_ids:

    print(f"\nProcessing conversation: {conversation_id}")

    conversation = dataset.filter(
        lambda x: x["conversation_id"] == conversation_id
    )

    customer_messages = []
    agent_messages = []

    for row in conversation:

        cleaned_message = clean_text(row["text"])

        if row["speaker"] == "client":

            customer_messages.append(cleaned_message)

        elif row["speaker"] == "agent":

            agent_messages.append(cleaned_message)


    # Combine customer messages
    customer_text = " ".join(customer_messages)

    # Combine agent messages
    agent_text = " ".join(agent_messages)


    # ==========================================
    # IDENTIFY ISSUE TYPE
    # ==========================================

    issue_types = []

    customer_lower = customer_text.lower()

    if (
        "dropped call" in customer_lower
        or "dropped calls" in customer_lower
    ):
        issue_types.append("Dropped calls")

    if (
        "slow" in customer_lower
        and "data" in customer_lower
    ):
        issue_types.append("Slow mobile data")

    if "poor reception" in customer_lower:
        issue_types.append("Poor reception")

    if "network" in customer_lower:
        issue_types.append("Network issue")


    issue_type = (
        ", ".join(issue_types)
        if issue_types
        else "Other"
    )


    # ==========================================
    # SENTIMENT
    # ==========================================

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

        if word in customer_lower:

            sentiment = "Negative"
            break


    # ==========================================
    # DEVICE DETECTION
    # ==========================================

    device_match = re.search(
        r"(iPhone\s+\d+(?:\s+Pro)?|Android|Samsung|Pixel)",
        customer_text,
        re.IGNORECASE
    )

    if device_match:

        device = device_match.group()

    else:

        device = "Not identified"


    # ==========================================
    # TROUBLESHOOTING
    # ==========================================

    troubleshooting_steps = extract_troubleshooting(
        agent_text
    )


    # ==========================================
    # OUTCOME
    # ==========================================

    outcome = determine_outcome(
        agent_text,
        troubleshooting_steps
    )


    # ==========================================
    # CREATE CASE
    # ==========================================

    case = {

        "case_id": conversation_id,

        "customer_problem": (
            customer_messages[0]
            if customer_messages
            else "Not available"
        ),

        # Text that will be used by retrieval
        "problem_text": customer_text,

        # Keep the original field as well
        "customer_text": customer_text,

        "device": device,

        "issue_type": issue_type,

        "sentiment": sentiment,

        "troubleshooting_steps": troubleshooting_steps,

        "outcome": outcome,

        "source_conversation": conversation_id
    }


    historical_cases.append(case)


# ==========================================
# SAVE JSON
# ==========================================

with open(
    "historical_cases.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        historical_cases,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("Historical cases created!")
print("==========================================")

print(
    "Number of cases:",
    len(historical_cases)
)


for case in historical_cases:

    print("\n------------------------------------------")

    print(
        "Case ID:",
        case["case_id"]
    )

    print(
        "Issue:",
        case["issue_type"]
    )

    print(
        "Device:",
        case["device"]
    )

    print(
        "Sentiment:",
        case["sentiment"]
    )

    print("Troubleshooting:")

    for step in case["troubleshooting_steps"]:

        print(" -", step)

    print(
        "Outcome:",
        case["outcome"]
    )


print("\nSaved as: historical_cases.json")

print("==========================================")