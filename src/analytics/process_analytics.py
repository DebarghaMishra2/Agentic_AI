import pandas as pd


# ============================================================
# BUILD EVENT-OBJECT TABLE
# ============================================================

def build_event_object(
    events: pd.DataFrame,
    relations: pd.DataFrame
) -> pd.DataFrame:

    event_columns = [
        "ocel:eid",
        "user",
        "cost_center",
        "ocel:activity",
        "ocel:timestamp",
        "location"
    ]

    event_columns = [
        col
        for col in event_columns
        if col in events.columns
    ]

    relation_columns = [
        "ocel:eid",
        "ocel:oid",
        "ocel:type"
    ]

    relation_columns = [
        col
        for col in relation_columns
        if col in relations.columns
    ]

    event_info = events[
        event_columns
    ].copy()

    relation_info = relations[
        relation_columns
    ].copy()

    event_object = relation_info.merge(
        event_info,
        on="ocel:eid",
        how="left"
    )

    return event_object


# ============================================================
# FILTER EVENTS BY OBJECT TYPE
# ============================================================

def get_events_by_object_type(
    event_object: pd.DataFrame,
    object_type: str
) -> pd.DataFrame:

    return event_object[
        event_object["ocel:type"]
        == object_type
    ].copy()


# ============================================================
# GET PR EVENTS
# ============================================================

def get_pr_events(
    event_object: pd.DataFrame
) -> pd.DataFrame:

    return get_events_by_object_type(
        event_object,
        "Purchase Requisition"
    )


# ============================================================
# GET PO EVENTS
# ============================================================

def get_po_events(
    event_object: pd.DataFrame
) -> pd.DataFrame:

    return get_events_by_object_type(
        event_object,
        "Purchase Order"
    )


# ============================================================
# GET INVOICE EVENTS
# ============================================================

def get_invoice_events(
    event_object: pd.DataFrame
) -> pd.DataFrame:

    return get_events_by_object_type(
        event_object,
        "Invoice"
    )


# ============================================================
# CALCULATE ACTIVITY DURATION
# ============================================================

def calculate_activity_duration(
    object_events: pd.DataFrame
) -> pd.DataFrame:

    df = object_events.sort_values(
        [
            "ocel:oid",
            "ocel:timestamp"
        ]
    ).copy()

    df["previous_activity"] = (
        df.groupby("ocel:oid")
        ["ocel:activity"]
        .shift(1)
    )

    df["previous_timestamp"] = (
        df.groupby("ocel:oid")
        ["ocel:timestamp"]
        .shift(1)
    )

    df["duration_hours"] = (
        df["ocel:timestamp"]
        - df["previous_timestamp"]
    ).dt.total_seconds() / 3600

    return df


# ============================================================
# CALCULATE TRANSITION STATISTICS
# ============================================================

def calculate_transition_statistics(
    object_events: pd.DataFrame
) -> pd.DataFrame:

    df = calculate_activity_duration(
        object_events
    )

    transitions = (
        df.dropna(
            subset=[
                "previous_activity",
                "duration_hours"
            ]
        )
        .groupby(
            [
                "previous_activity",
                "ocel:activity"
            ]
        )
        .agg(
            occurrences=(
                "duration_hours",
                "count"
            ),
            average_hours=(
                "duration_hours",
                "mean"
            ),
            median_hours=(
                "duration_hours",
                "median"
            ),
            p90_hours=(
                "duration_hours",
                lambda x: x.quantile(0.90)
            ),
            max_hours=(
                "duration_hours",
                "max"
            )
        )
        .reset_index()
    )

    transitions = transitions.sort_values(
        "average_hours",
        ascending=False
    )

    return transitions


# ============================================================
# PR CYCLE TIME
# ============================================================

def calculate_pr_cycle_time(
    pr_events: pd.DataFrame
) -> pd.DataFrame:

    duration = (
        pr_events
        .groupby("ocel:oid")
        .agg(
            start_time=(
                "ocel:timestamp",
                "min"
            ),
            end_time=(
                "ocel:timestamp",
                "max"
            )
        )
        .reset_index()
    )

    duration["cycle_time_hours"] = (
        duration["end_time"]
        - duration["start_time"]
    ).dt.total_seconds() / 3600

    return duration


# ============================================================
# PR APPROVAL TIME
# ============================================================

def calculate_pr_approval_time(
    pr_events: pd.DataFrame
) -> pd.DataFrame:

    created = (
        pr_events[
            pr_events["ocel:activity"]
            == "Create PR"
        ]
        .groupby("ocel:oid")
        ["ocel:timestamp"]
        .min()
    )

    approved = (
        pr_events[
            pr_events["ocel:activity"]
            == "Approve PR"
        ]
        .groupby("ocel:oid")
        ["ocel:timestamp"]
        .min()
    )

    rejected = (
        pr_events[
            pr_events["ocel:activity"]
            == "Reject PR"
        ]
        .groupby("ocel:oid")
        ["ocel:timestamp"]
        .min()
    )

    result = pd.concat(
        [
            created,
            approved,
            rejected
        ],
        axis=1
    )

    result.columns = [
        "create_time",
        "approve_time",
        "reject_time"
    ]

    result["approval_time_hours"] = (
        result["approve_time"]
        - result["create_time"]
    ).dt.total_seconds() / 3600

    result["rejection_time_hours"] = (
        result["reject_time"]
        - result["create_time"]
    ).dt.total_seconds() / 3600

    return result.reset_index()


# ============================================================
# REJECTION METRICS
# ============================================================

def calculate_rejection_metrics(
    pr_events: pd.DataFrame
) -> dict:

    created = (
        pr_events[
            pr_events["ocel:activity"]
            == "Create PR"
        ]["ocel:oid"]
        .nunique()
    )

    approved = (
        pr_events[
            pr_events["ocel:activity"]
            == "Approve PR"
        ]["ocel:oid"]
        .nunique()
    )

    rejected = (
        pr_events[
            pr_events["ocel:activity"]
            == "Reject PR"
        ]["ocel:oid"]
        .nunique()
    )

    rejection_rate = (
        rejected / created * 100
        if created > 0
        else 0
    )

    approval_rate = (
        approved / created * 100
        if created > 0
        else 0
    )

    return {
        "created_prs": int(created),
        "approved_prs": int(approved),
        "rejected_prs": int(rejected),
        "approval_rate": float(approval_rate),
        "rejection_rate": float(rejection_rate)
    }


# ============================================================
# THREE-WAY MATCH METRICS
# ============================================================

def calculate_match_metrics(
    events: pd.DataFrame
) -> dict:

    successful = (
        events["ocel:activity"]
        == "Three-way Match - Successful"
    ).sum()

    mismatch = (
        events["ocel:activity"]
        == "Three-way Match - Mismatch Found"
    ).sum()

    total = successful + mismatch

    mismatch_rate = (
        mismatch / total * 100
        if total > 0
        else 0
    )

    success_rate = (
        successful / total * 100
        if total > 0
        else 0
    )

    return {
        "total_matches": int(total),
        "successful_matches": int(successful),
        "mismatches": int(mismatch),
        "success_rate": float(success_rate),
        "mismatch_rate": float(mismatch_rate)
    }


# ============================================================
# PROCESS KPI SUMMARY
# ============================================================

def generate_process_kpis(
    events: pd.DataFrame,
    pr_events: pd.DataFrame
) -> dict:

    cycle_time = calculate_pr_cycle_time(
        pr_events
    )

    rejection = calculate_rejection_metrics(
        pr_events
    )

    match = calculate_match_metrics(
        events
    )

    return {

        "total_prs":
            rejection["created_prs"],

        "approved_prs":
            rejection["approved_prs"],

        "rejected_prs":
            rejection["rejected_prs"],

        "approval_rate":
            rejection["approval_rate"],

        "rejection_rate":
            rejection["rejection_rate"],

        "average_cycle_time_hours":
            cycle_time[
                "cycle_time_hours"
            ].mean(),

        "median_cycle_time_hours":
            cycle_time[
                "cycle_time_hours"
            ].median(),

        "p90_cycle_time_hours":
            cycle_time[
                "cycle_time_hours"
            ].quantile(0.90),

        "three_way_match_success_rate":
            match["success_rate"],

        "three_way_match_mismatch_rate":
            match["mismatch_rate"]
    }