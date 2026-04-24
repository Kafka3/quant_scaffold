def get_costs(config: dict) -> tuple[float, float]:
    return float(config.get("fees", 0.0)), float(config.get("slippage", 0.0))
