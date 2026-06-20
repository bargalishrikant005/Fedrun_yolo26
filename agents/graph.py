"""
agents/graph.py
FedASIO-YOLO26: LangGraph StateGraph — Multi-Agent Pipeline Orchestration
Wires all 8 agents into a directed graph with conditional routing and error handling.
"""
import logging
from langgraph.graph import StateGraph, END
from agents.state import FedASIOState
from agents.data_agent import data_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.augmentation_agent import augmentation_agent
from agents.asio_agent import asio_agent
from agents.segmentation_agent import segmentation_agent
from agents.evaluation_agent import evaluation_agent
from agents.xai_agent import xai_agent
from agents.report_agent import report_agent

logger = logging.getLogger(__name__)


def route_on_error(state: FedASIOState) -> str:
    """Conditional edge: if any agent set an error, go to END."""
    if state.get("error"):
        logger.error(f"Pipeline stopped at {state.get('error_agent')}: {state['error']}")
        return "end"
    return "continue"


def route_by_stage(state: FedASIOState) -> str:
    """Route based on FL stage: training vs inference."""
    stage = state.get("fl_stage", "infer")
    return "train" if stage == "train" else "infer"


def route_after_evaluation(state: FedASIOState) -> str:
    """Route: if stage is evaluate or test, go to END. Otherwise (infer mode), do XAI and Report."""
    if state.get("error"):
        logger.error(f"Pipeline stopped at {state.get('error_agent')}: {state['error']}")
        return "end"
    stage = state.get("fl_stage", "infer")
    if stage in ["evaluate", "test"]:
        return "end"
    return "continue"


def build_training_graph() -> StateGraph:
    """
    Training pipeline graph:
    DataAgent → PreprocessAgent → AugmentAgent → ASIOAgent → SegmentAgent → EvalAgent → END
    """
    builder = StateGraph(FedASIOState)

    builder.add_node("data_agent", data_agent)
    builder.add_node("preprocessing_agent", preprocessing_agent)
    builder.add_node("augmentation_agent", augmentation_agent)
    builder.add_node("asio_agent", asio_agent)
    builder.add_node("segmentation_agent", segmentation_agent)
    builder.add_node("evaluation_agent", evaluation_agent)

    builder.set_entry_point("data_agent")

    # Linear chain with error checking at each step
    builder.add_conditional_edges("data_agent",
        route_on_error, {"end": END, "continue": "preprocessing_agent"})
    builder.add_conditional_edges("preprocessing_agent",
        route_on_error, {"end": END, "continue": "augmentation_agent"})
    builder.add_conditional_edges("augmentation_agent",
        route_on_error, {"end": END, "continue": "asio_agent"})
    builder.add_conditional_edges("asio_agent",
        route_on_error, {"end": END, "continue": "segmentation_agent"})
    builder.add_conditional_edges("segmentation_agent",
        route_on_error, {"end": END, "continue": "evaluation_agent"})
    builder.add_edge("evaluation_agent", END)

    return builder.compile()


def build_inference_graph() -> StateGraph:
    """
    Inference pipeline graph:
    DataAgent → PreprocessAgent → SegmentAgent → EvalAgent → XAIAgent → ReportAgent → END
    """
    builder = StateGraph(FedASIOState)

    builder.add_node("data_agent", data_agent)
    builder.add_node("preprocessing_agent", preprocessing_agent)
    builder.add_node("segmentation_agent", segmentation_agent)
    builder.add_node("evaluation_agent", evaluation_agent)
    builder.add_node("xai_agent", xai_agent)
    builder.add_node("report_agent", report_agent)

    builder.set_entry_point("data_agent")

    builder.add_conditional_edges("data_agent",
        route_on_error, {"end": END, "continue": "preprocessing_agent"})
    builder.add_conditional_edges("preprocessing_agent",
        route_on_error, {"end": END, "continue": "segmentation_agent"})
    builder.add_conditional_edges("segmentation_agent",
        route_on_error, {"end": END, "continue": "evaluation_agent"})
    builder.add_conditional_edges("evaluation_agent",
        route_after_evaluation, {"end": END, "continue": "xai_agent"})
    builder.add_conditional_edges("xai_agent",
        route_on_error, {"end": END, "continue": "report_agent"})
    builder.add_edge("report_agent", END)

    return builder.compile()


# Pre-compiled graphs (reusable)
TRAINING_GRAPH = None
INFERENCE_GRAPH = None


def get_training_graph():
    global TRAINING_GRAPH
    if TRAINING_GRAPH is None:
        TRAINING_GRAPH = build_training_graph()
    return TRAINING_GRAPH


def get_inference_graph():
    global INFERENCE_GRAPH
    if INFERENCE_GRAPH is None:
        INFERENCE_GRAPH = build_inference_graph()
    return INFERENCE_GRAPH
