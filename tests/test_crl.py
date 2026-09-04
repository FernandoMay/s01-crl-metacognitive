"""
Tests for Cognitive Resilience Layer Simulation
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crl_simulation import (
    Agent, AgentState, FailureType, DAGNode, ExecutionDAG,
    MetacognitiveObserver, MetacognitiveEvaluator, MetacognitiveModifier,
    CognitiveResilienceLayer, SimulationRunner
)


class TestAgent:
    def test_agent_creation(self):
        agent = Agent(agent_id=0, capabilities=["compute", "storage"])
        assert agent.agent_id == 0
        assert agent.state == AgentState.IDLE

    def test_health_score(self):
        agent = Agent(agent_id=0, capabilities=["compute"])
        agent.task_load = 0.5
        agent.memory_usage = 0.3
        score = agent.health_score
        assert 0 <= score <= 1

    def test_failed_agent_health(self):
        agent = Agent(agent_id=0, capabilities=["compute"])
        agent.state = AgentState.FAILED
        assert agent.health_score == 0.0


class TestDAGNode:
    def test_node_creation(self):
        node = DAGNode(node_id=0, agent_id=0, task_name="task_0")
        assert node.status == "pending"


class TestExecutionDAG:
    def test_ready_nodes(self):
        nodes = [
            DAGNode(node_id=0, agent_id=0, task_name="t0"),
            DAGNode(node_id=1, agent_id=0, task_name="t1", dependencies=[0]),
        ]
        dag = ExecutionDAG(dag_id="test", nodes=nodes, edges=[(0, 1)])
        ready = dag.get_ready_nodes(set())
        assert len(ready) == 1
        assert ready[0].node_id == 0

    def test_dag_completion(self):
        nodes = [DAGNode(node_id=0, agent_id=0, task_name="t0")]
        nodes[0].status = "completed"
        dag = ExecutionDAG(dag_id="test", nodes=nodes, edges=[])
        assert dag.is_complete() is True


class TestMetacognitiveObserver:
    def test_observe(self):
        observer = MetacognitiveObserver()
        agents = [Agent(agent_id=i, capabilities=["compute"]) for i in range(3)]
        nodes = [DAGNode(node_id=0, agent_id=0, task_name="t0")]
        dag = ExecutionDAG(dag_id="test", nodes=nodes, edges=[])
        state = observer.observe(agents, dag)
        assert "timestamp" in state
        assert "health_scores" in state

    def test_anomaly_detection(self):
        observer = MetacognitiveObserver()
        agents = [Agent(agent_id=i, capabilities=["compute"]) for i in range(3)]
        nodes = [DAGNode(node_id=0, agent_id=0, task_name="t0")]
        dag = ExecutionDAG(dag_id="test", nodes=nodes, edges=[])
        observer.observe(agents, dag)
        agents[0].state = AgentState.FAILED
        observer.observe(agents, dag)
        anomaly = observer.detect_anomaly(observer.state_history[-1])
        assert anomaly > 0


class TestMetacognitiveEvaluator:
    def test_evaluate(self):
        observer = MetacognitiveObserver()
        evaluator = MetacognitiveEvaluator()
        agents = [Agent(agent_id=i, capabilities=["compute"]) for i in range(3)]
        nodes = [DAGNode(node_id=0, agent_id=0, task_name="t0")]
        dag = ExecutionDAG(dag_id="test", nodes=nodes, edges=[])
        observer.observe(agents, dag)
        observer.observe(agents, dag)
        evaluation = evaluator.evaluate(observer, agents, dag)
        assert "needs_intervention" in evaluation
        assert "recommended_actions" in evaluation


class TestCognitiveResilienceLayer:
    def test_crl_creation(self):
        crl = CognitiveResilienceLayer()
        assert crl.cycle_count == 0

    def test_run_cycle(self):
        crl = CognitiveResilienceLayer()
        agents = [Agent(agent_id=i, capabilities=["compute"]) for i in range(3)]
        nodes = [DAGNode(node_id=0, agent_id=0, task_name="t0")]
        dag = ExecutionDAG(dag_id="test", nodes=nodes, edges=[])
        result = crl.run_cycle(agents, dag)
        assert "cycle" in result
        assert "evaluation" in result


class TestSimulationRunner:
    def test_runner_creation(self):
        runner = SimulationRunner(num_agents=5, num_tasks=10)
        assert runner.num_agents == 5

    def test_create_agents(self):
        runner = SimulationRunner(num_agents=5, num_tasks=10)
        agents = runner.create_agents()
        assert len(agents) == 5

    def test_create_dag(self):
        runner = SimulationRunner(num_agents=5, num_tasks=10)
        agents = runner.create_agents()
        dag = runner.create_dag(agents)
        assert len(dag.nodes) == 10

    def test_experiment_run(self):
        runner = SimulationRunner(num_agents=5, num_tasks=10)
        results = runner.run_experiment()
        assert "completion_rate" in results
        assert "total_modifications" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
