import re


# ============================================================
# CUSTOMER COMPLAINT
# ============================================================

query = (
    "My calls keep dropping and my mobile internet "
    "is very slow."
)


# ============================================================
# ANALYZE CUSTOMER COMPLAINT
# ============================================================

def analyze_complaint(text):

    text_lower = text.lower()


    # --------------------------------------------------------
    # CATEGORY / INTENT
    # --------------------------------------------------------

    categories = []

    if (
        "drop" in text_lower
        and "call" in text_lower
    ):
        categories.append("Dropped calls")

    if (
        "slow" in text_lower
        and (
            "data" in text_lower
            or "internet" in text_lower
        )
    ):
        categories.append("Slow mobile data")

    if (
        "poor reception" in text_lower
        or "weak signal" in text_lower
    ):
        categories.append("Poor network reception")


    if categories:

        intent = "Network connectivity issue"

    else:

        intent = "General support issue"


    # --------------------------------------------------------
    # PRODUCT / SERVICE
    # --------------------------------------------------------

    has_voice = (
        "call" in text_lower
        or "calls" in text_lower
    )

    has_data = (
        "data" in text_lower
        or "internet" in text_lower
    )


    if has_voice and has_data:

        product = "Mobile voice and data"

    elif has_voice:

        product = "Mobile voice service"

    elif has_data:

        product = "Mobile data service"

    else:

        product = "Telecom service"


    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    negative_words = [
        "frustrated",
        "frustrating",
        "angry",
        "annoying",
        "terrible",
        "disappointed",
        "unhappy",
        "problem",
        "issue",
        "not working",
        "can't",
        "cannot"
    ]


    negative_count = 0

    for word in negative_words:

        if word in text_lower:

            negative_count += 1


    if negative_count >= 2:

        sentiment = "Negative"

    elif negative_count == 1:

        sentiment = "Slightly negative"

    else:

        sentiment = "Neutral"


    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    high_severity_words = [
        "emergency",
        "completely down",
        "no service",
        "cannot make calls",
        "unable to call",
        "all day"
    ]


    medium_severity_words = [
        "keep dropping",
        "repeatedly",
        "very slow",
        "frequent",
        "constant",
        "keeps"
    ]


    severity = "Low"


    for word in high_severity_words:

        if word in text_lower:

            severity = "High"
            break


    if severity == "Low":

        for word in medium_severity_words:

            if word in text_lower:

                severity = "Medium"
                break


    # --------------------------------------------------------
    # RETURN ANALYSIS
    # --------------------------------------------------------

    return {

        "intent": intent,

        "category": categories
        if categories
        else ["Other"],

        "product": product,

        "severity": severity,

        "sentiment": sentiment
    }


# ============================================================
# RUN ANALYSIS
# ============================================================

analysis = analyze_complaint(query)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("       CUSTOMER COMPLAINT ANALYSIS")
print("==========================================")


print("\nCustomer Complaint:")
print(query)


print("\nIntent:")
print(analysis["intent"])


print("\nCategory:")

for category in analysis["category"]:

    print(" -", category)


print("\nProduct:")
print(analysis["product"])


print("\nSeverity:")
print(analysis["severity"])


print("\nSentiment:")
print(analysis["sentiment"])


print("\n==========================================")
print("       ANALYSIS COMPLETED")
print("==========================================")