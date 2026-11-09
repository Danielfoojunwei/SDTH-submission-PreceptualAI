"""Mission wordings for testing the intent reader.

For each known mission: its original intent plus four paraphrases (two used
for distillation, two held out). NOVEL missions never appear in distillation:
their goals and weights differ from every training mission, so they test
whether the reader generalises beyond the wordings it was adapted on.
"""
from .missions import MISSIONS, WAREHOUSE_MISSIONS, Mission, _w

PARAPHRASES = {
    "protect_video": [
        "Car-03's inspection stream is the priority. Its video must stay at full rate; the other cars can slow down.",
        "We are watching car-03's live camera. Do whatever keeps car-03's uplink intact, even at the expense of the rest.",
        "car-03 carries the only live video feed right now; keep it at full quality and accept slower links elsewhere.",
        "Top priority is car-03's camera uplink. Other cars losing throughput is acceptable.",
    ],
    "convoy_continuity": [
        "Cars are driving in a close convoy. Do not reconnect anyone or change channels; steady links beat speed.",
        "During the convoy run, keep every car on its current access point. Interruptions are worse than slow links.",
        "Convoy manoeuvre in progress: no roaming, no channel changes, uninterrupted connectivity first.",
        "Keep links stable for the convoy. Avoid any BSSID switches even if throughput suffers.",
    ],
    "bulk_upload": [
        "All cars are uploading survey data. Push as much total data as possible; short reconnections are fine.",
        "Maximise the fleet's aggregate upload rate for the survey maps. Brief link drops are acceptable.",
        "Survey upload window: total delivered data is what counts, roaming is allowed.",
        "Get the most data off the cars overall. Reconnecting cars to better access points is fine.",
    ],
    "fair_share": [
        "Each car runs the same patrol, so share capacity evenly and do not let any car starve.",
        "Fairness matters most: every car should get a similar share of the network.",
        "No car may be left behind. Spread bandwidth evenly across the patrol fleet.",
        "Balance capacity so that the worst-served car is as well served as possible.",
    ],
    "spectrum_etiquette": [
        "Protected lab equipment uses some channels. Keep the cars off those channels and limit co-channel interference.",
        "Be a good spectrum neighbour: avoid the protected channels and keep interference low.",
        "Some channels belong to protected instruments next door. Stay clear of them and keep the air clean.",
        "Minimise use of protected channels and co-channel interference, even if throughput drops.",
    ],
    "teleop_assist": [
        "amr-017 is being driven remotely. Its video uplink has to stay at full rate, whatever it costs the others.",
        "An operator is teleoperating amr-017. Keep amr-017's camera feed intact above everything else.",
        "Remote control session on amr-017: protect its video link first, other robots can lose throughput.",
        "amr-017's teleop video must not degrade. Other robots are lower priority.",
    ],
    "peak_picking": [
        "Peak wave: robots are moving fast. Keep all safety links up and do not reconnect robots during the wave.",
        "During the picking peak, avoid roaming and make sure every safety heartbeat stays above its floor.",
        "Busy picking wave: stable links and intact safety links matter more than throughput.",
        "No reconnections during peak picking, and every robot's safety link must stay healthy.",
    ],
    "map_rollout": [
        "We are pushing a new site map to every robot. Maximise total delivered data; short reconnections are fine.",
        "Map rollout: get the update out as fast as possible across the fleet, roaming allowed.",
        "Total data delivered to the fleet is what matters during the map push. Brief link drops are acceptable.",
        "Deliver the new map to all robots quickly; maximise aggregate throughput.",
    ],
    "shift_fairness": [
        "All robots carry equal task load this shift, so share capacity evenly and starve no one.",
        "Even distribution of network capacity across robots is the goal for this shift.",
        "No robot should be left with a poor link; balance capacity across the fleet.",
        "Fair share for every robot this shift: protect the worst-served robot.",
    ],
    "scanner_coexistence": [
        "Handheld scanners share some channels. Keep robots away from scanner channels and keep interference low.",
        "Barcode scanners need their channels. Move robots off them and avoid co-channel interference.",
        "Coexist with the scanners: stay off their channels and keep the airwaves clean.",
        "Avoid scanner-occupied channels and minimise interference, even at some cost in throughput.",
    ],
}
TRAIN_PARAPHRASES = 2   # first two of each list; the last two are held out

NOVEL = {
    "safety_drill": (Mission("safety_drill", "", _w(viol=4.0, U=0.3, V=0.5, ho=0.5)),
                     ["Safety drill: what matters is that no robot loses its safety heartbeat. Throughput is irrelevant.",
                      "During the drill every safety link must stay above its minimum; ignore data rates.",
                      "Keep all safety-critical links healthy for the drill. Total throughput does not matter."]),
    "rf_quiet": (Mission("rf_quiet", "", _w(prot=5.0, W=2.0, B=1.5, ho=4.0, U=0.3)),
                 ["The RF team is measuring. Keep robots on clean channels away from scanners and do not move them between access points.",
                  "RF survey in progress: no roaming, no scanner channels, as little interference as possible.",
                  "Quiet airwaves please: stay off protected channels, keep interference low, and avoid reconnections."]),
    "inspect_amr041": (Mission("inspect_amr041", "", _w(focus=4.0, fair=1.5, U=0.3), ("amr-041",)),
                       ["amr-041 is being inspected remotely. Its camera must not drop, but no other robot should be starved.",
                        "Keep amr-041's video link at full rate while still sharing capacity fairly among the rest.",
                        "Remote inspection of amr-041: protect its stream, and keep the other robots evenly served."]),
    "night_restock": (Mission("night_restock", "", _w(U=2.0, V=1.5, ho=0.1, fair=0.3)),
                      ["Overnight restock: every robot must finish its uploads and total data matters; reconnections are fine.",
                       "Night shift: maximise delivered data and give each robot the rate it needs. Roaming is allowed.",
                       "Restocking overnight: get all uploads through quickly, both overall and per robot. Switching access points is fine."]),
}


def all_known() -> dict[str, Mission]:
    return {**MISSIONS, **WAREHOUSE_MISSIONS}
