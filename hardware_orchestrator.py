"""
HackerRank Task Definition

This file contains:
- PROMPT: The challenge prompt given to the agent
- TOOLS: The tool definitions available to the agent
- TOOL_HANDLERS: The tool handler functions
- grading_func: Function that validates the agent's answer
"""

from collections.abc import Callable
import json 
from typing import Any, TypedDict #lets you describe the shape of a dictionary.

from anthropic.types import ToolUnionParam


# class PythonExpressionToolResult(TypedDict):
#     result: Any
#     error: str | None

class SimulateResult(TypedDict):
    ok: bool
    fits: bool
    estimated_seconds: float
    dollars: float
    notes: str

class SubmitResult(TypedDict):
    answer: Any
    submitted: bool

hardware_options = {
    "CPU": {"throughput": 1.0, "vram_gb": 999.0, "cost_per_hour": 0.20, "startup_time": 0.0},
    "T4": {"throughput": 6.0, "vram_gb": 16.0, "cost_per_hour": 0.60, "startup_time": 0.0},
    "TPU": {"throughput":16.0, "vram_gb": 32.0, "cost_per_hour": 1.80, "startup_time": 120.0}
}
# throughput: how fast it is
# vram_gb: how much memory it has
# cost_per_hour: how expensive it is
# startup_time: extra waiting time (TPU has compile overhead)

memory_savings = {"fp16": 1.0, "bf16": 1.0, "int8":0.55}

# fp16 uses normal memory
# bf16 uses normal memory
# int8 uses less memory (55% of fp16 memory)

memory_per_sequence_gb = 0.15 #this is the assumed memory cost for processing a single unit of data,.

job_scenarios= {
    "scenario_1":{
        "title": "The task is too slow on a CPU(E.g on Colab)",
        "weights_gb_fp16": 4.0,
        "optimizer_gb_fp16": 4.0,
        "sequence": 2048,
        "batch": 8,
        "steps": 600,
        "target_seconds": 900.0,
        "must_change": False,
    },
    "scenario_2":{
        "title": "GPU goes out of memory unless you adjust batch or quantize",
        "weights_gb_fp16": 12.0,
        "optimizer_gb_fp16": 12.0,
        "sequence": 4096,
        "batch": 16,
        "steps": 500,
        "target_seconds": 900.0,
        "must_change": True,
    },
    "scenario_3":{
        "title": "TPU compile overhead",
        "weights_gb_fp16": 10.0,
        "optimizer_gb_fp16": 10.0,
        "sequence": 4096,
        "batch": 8,
        "steps": 3000,
        "target_seconds": 1200.0,
        "must_change": False
    },
}
#these are different machine learning training scenarios with their resource requirements and constraints.
def calculate_memory_needed(model_weights:float,optimizer_mem: float, batch:int, sequence:int, precision:str) -> float:
    memory_factor = memory_savings.get(precision)
    if memory_factor is None:
        return float("inf")
    weights = model_weights * memory_factor
    opt = optimizer_mem * memory_factor
    activation = memory_per_sequence_gb * (batch * (sequence / 2048.0))
    return weights + opt + activation
#total_memory = model weights + optimizer memory + activation memory
#weights memory depends on precision
# optimizer memory depends on precision
# activation memory depends on batch and sequence length
# Then it returns the total memory needed in GB.
# If memory needed > device vram → OOM (out of memory)


def estimate_training_time(steps:int, device:str, batch:int, sequence:int) -> float:
    options = hardware_options[device]
    work = steps * (batch * (sequence/2048.0))
    return options["startup_time"] + (work / options["throughput"])
#takes the number of steps, device type, batch size, and sequence length as input.
# It calculates the total amount of work to be done based on these parameters.
# Then it estimates the training time by dividing the work by the device's throughput and adding any startup time.
# It returns the estimated training time in seconds.

def simulate(plan: Any) -> SimulateResult:
    """
    plan format: dictionary with keys
    - "scenario": "scenario_1"|"scenario_2"|"scenario_3",
    - "device": "CPU"|"T4"|"TPU",
    - "batch": int,
    - "precision": "fp16"|"bf16"|"int8"
    """
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return {"ok": False, "fits": False, "estimated_seconds":1e18, "dollars":1e18, "notes": "bad_json"}
    if not isinstance(plan, dict):
        return {"ok": False, "fits": False, "estimated_seconds": 1e18, "dollars": 1e18, "notes": "not_a_dict"}
    scenario = plan.get("scenario")
    device = plan.get("device")
    batch = plan.get("batch")
    precision = plan.get("precision")

    if scenario not in job_scenarios:
        return{"ok": False, "fits": False, "estimated_seconds": 1e18, "dollars":1e18, "notes": "unknown_scenario"}
    if device not in hardware_options:
         return{"ok": False, "fits": False, "estimated_seconds": 1e18, "dollars":1e18, "notes": "unknown_device"}
    if not isinstance (batch, int) or batch <= 0 or batch > 64:
        return{"ok": False, "fits": False, "estimated_seconds": 1e18, "dollars":1e18, "notes": "bad_batch"}
    if precision not in memory_savings:
        return{"ok": False, "fits": False, "estimated_seconds": 1e18, "dollars":1e18, "notes": "bad_precision"}
    s = job_scenarios[scenario]
    memory = calculate_memory_needed(s["weights_gb_fp16"],s["optimizer_gb_fp16"], batch,  s["sequence"], precision )
    fits = memory <= hardware_options[device]["vram_gb"]
    estimated_seconds = estimate_training_time(s["steps"], device, batch, s["sequence"])
    dollars = (estimated_seconds/3600.0) * hardware_options[device]["cost_per_hour"]
    notes = f"mem_gb={memory:.2f}, vram_gb={hardware_options[device]['vram_gb']:.0f}"

    return {"ok": True, "fits": fits, "estimated_seconds": float(estimated_seconds), "dollars": float(dollars), "notes": notes}
#What simulate does:
# If plan is JSON text, turn it into a dictionary
# Check if scenario/device/batch/precision are valid(error cases)
# Compute memory needed
# Compute if it fits in VRAM
# Compute runtime seconds
# Compute cost dollars
# Return everything in a result dictionary
# That’s it


def submit_answer_tool(answer: Any) -> SubmitResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}
#This doesn’t grade anything.
#It simply packages your final answer 

# Tool definitions for Anthropic API
TOOLS: list[ToolUnionParam] = [
    {
        "name": "simulate",
        "description": "Simulate a plan (device/precision/batch) for a scenario",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "description": "plan dict or json string with keys: scenario, device, batch, precision."
                }
            },
            "required": ["plan"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit the final plan for scenario_1 to scenario_3 as Json",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {}},
            "required": ["answer"],
        },
    },
]
#Tells the Anthropic API about the tools available to the agent.


# Tool handlers mapping
TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "simulate": simulate,
    "submit_answer": submit_answer_tool,
}


# The challenge prompt
PROMPT = f"""
You are an ML infrastructure engineer. Your task is to select 1 valid configuration per scenario.
Rules:
- Scenario 2: You MUST change batch or precision from default (16, fp16).
- A valid plan returns fits: True and estimated_seconds <= target_seconds.

Scenarios:
{json.dumps(job_scenarios, indent=2)}
""".strip()


# Grading function - validates the agent's submitted answer

def _parse_plan(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None

def grading_func(result: Any) -> bool:
    """
    Validates the agent's answer.

    Args:
        result: The value returned from run_agent_loop (typically the submitted answer)

    Returns:
        True if the answer is correct, False otherwise
    """
    plan = _parse_plan(result)
    if not plan or set(plan.keys()) != {"scenario_1", "scenario_2", "scenario_3"}:
        return False
    for sid, cfg in plan.items():
        if not isinstance(cfg, dict):
            return False

        test_plan = {
            "scenario": sid,
            "device": cfg.get("device"),
            "precision": cfg.get("precision"),
            "batch": cfg.get("batch")
        }
        output = simulate(test_plan)
        if not output["ok"] or not output["fits"]:
            return False
        if output["estimated_seconds"] > job_scenarios[sid]["target_seconds"]:
            return False
        
        if job_scenarios[sid]["must_change"]:
            if (
               cfg.get("batch") == job_scenarios[sid]["batch"]
               and cfg.get("precision") == "fp16"
            ):
                return False
#this part says:
# For scenario_2:
# - S2 has `must_change=True`, default is batch=16 + fp16
# - If you submit batch=16 + fp16 → `False`
# - Change one (batch or precision) → ok
    return True
# This function checks if the submitted answer meets all the requirements for each scenario.
