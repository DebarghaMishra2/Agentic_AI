from pathlib import Path
import pandas as pd
import pm4py


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)


# ============================================================
# DATABASE PATH
# ============================================================

DB_PATH = (
    BASE_DIR
    / "data"
    / "lrms_02_p2p.sqlite"
)


# ============================================================
# CHECK DATABASE
# ============================================================

def check_database():

    print("Looking for database at:")
    print(DB_PATH)

    if not DB_PATH.exists():

        raise FileNotFoundError(
            f"Database not found at:\n{DB_PATH}"
        )

    return DB_PATH


# ============================================================
# LOAD OCEL
# ============================================================

def load_ocel():

    database_path = check_database()

    ocel = pm4py.read_ocel2(
        str(database_path)
    )

    return ocel


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_all():

    ocel = load_ocel()

    events = ocel.events.copy()

    relations = ocel.relations.copy()

    objects = ocel.objects.copy()


    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    if "ocel:timestamp" in events.columns:

        events["ocel:timestamp"] = pd.to_datetime(
            events["ocel:timestamp"],
            utc=True
        )


    if "ocel:timestamp" in relations.columns:

        relations["ocel:timestamp"] = pd.to_datetime(
            relations["ocel:timestamp"],
            utc=True
        )


    return (
        events,
        relations,
        objects
    )


# ============================================================
# DATABASE INFORMATION
# ============================================================

def database_info():

    events, relations, objects = load_all()

    return {

        "database": str(DB_PATH),

        "events": len(events),

        "relations": len(relations),

        "objects": len(objects),

        "event_columns":
            list(events.columns),

        "relation_columns":
            list(relations.columns),

        "object_columns":
            list(objects.columns)
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    info = database_info()

    print("\n===================================")
    print("       P2P OCEL DATABASE")
    print("===================================\n")

    print(
        f"Database : {info['database']}"
    )

    print(
        f"Events   : {info['events']}"
    )

    print(
        f"Relations: {info['relations']}"
    )

    print(
        f"Objects  : {info['objects']}"
    )

    print("\nEvent columns:")
    print(info["event_columns"])

    print("\nRelation columns:")
    print(info["relation_columns"])

    print("\nObject columns:")
    print(info["object_columns"])