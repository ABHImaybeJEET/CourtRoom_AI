import pytest
from graph.courtroom_graph import create_courtroom_graph, should_continue, CourtRoomState


def test_create_courtroom_graph():
    graph = create_courtroom_graph()
    assert graph is not None
    # Verify nodes are registered
    nodes = graph.nodes
    assert "prosecution_round" in nodes
    assert "defense_round" in nodes
    assert "judge_evaluation" in nodes
    assert "jury_simulation" in nodes
    assert "retrieve_context" in nodes


def test_should_continue_logic():
    # When current round is 0 and max_rounds is 2, it should increment round
    state_round_0: CourtRoomState = {"round_number": 0, "max_rounds": 2}
    assert should_continue(state_round_0) == "increment_round"

    # When current round is 1 and max_rounds is 2, it should proceed to jury_simulation
    state_round_1: CourtRoomState = {"round_number": 1, "max_rounds": 2}
    assert should_continue(state_round_1) == "jury_simulation"

    # When max_rounds is 1, even round 0 goes straight to jury_simulation
    state_single_round: CourtRoomState = {"round_number": 0, "max_rounds": 1}
    assert should_continue(state_single_round) == "jury_simulation"
