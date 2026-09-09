from ..config import APPROACHING_TARGET_MONTHS, HIGH_PROBABILITY, LOW_PROGRESS, SLOW_PROGRESS_VELOCITY, SPEND_PROGRESS_GAP
def generate_warnings(data: dict, cost_probability: float | None = None, time_probability: float | None = None) -> list[str]:
    warnings = []
    def add(message): warnings.append(message)
    if cost_probability is not None and cost_probability >= HIGH_PROBABILITY: add("High probability of cost overrun.")
    if time_probability is not None and time_probability >= HIGH_PROBABILITY: add("High probability of schedule overrun.")
    if data.get("progress_gap") is not None and float(data["progress_gap"]) >= SPEND_PROGRESS_GAP: add("Expenditure is significantly ahead of physical progress.")
    if data.get("months_to_target_completion") is not None:
        months_to_target = float(data["months_to_target_completion"])

        if months_to_target < 0:
            add("Target completion date has already passed.")

        elif (
            data.get("physical_progress") is not None
            and months_to_target <= APPROACHING_TARGET_MONTHS
            and float(data["physical_progress"]) < LOW_PROGRESS
        ):
            add("Physical progress may be insufficient to meet the target completion date.")
    if data.get("progress_velocity") is not None and float(data["progress_velocity"]) <= SLOW_PROGRESS_VELOCITY: add("Project progress has shown limited recent movement.")
    return warnings
