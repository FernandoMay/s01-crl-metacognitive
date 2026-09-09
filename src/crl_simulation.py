"""
Cognitive Resilience Layer as a Metacognitive Control Plane
for Distributed AI Systems

Paper: Cognitive Resilience Layer as a Metacognitive Control Plane
       for Distributed AI Systems
Venue: WSSE 2026
Authors: Fernando May et al.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import random
import time
from enum import Enum
import copy


class AgentState(Enum):
    IDLE = 0
    EXECUTING = 1
    FAILED = 2
    RECOVERING = 3
    TERMINATED = 4


class FailureType(Enum):
    NONE = 0
    TIMEOUT = 1
    MEMORY_OVERFLOW = 2
    NETWORK_PARTITION = 3
    COMPUTATION_ERROR = 4
    CASCADE = 5


@dataclass
class Agent:
    agent_id: int
    capabilities: List[str]
    state: AgentState = AgentState.IDLE
    task_load: float = 0.0
    memory_usage: float = 0.0
    failure_count: int = 0
    last_heartbeat: float = 0.0
    neighbors: List[int] = field(default_factory=list)

    @property
    def health_score(self) -> float:
        if self.state == AgentState.FAILED:
            return 0.0
        load_penalty = self.task_load * 0.3
        memory_penalty = self.memory_usage * 0.2
        failure_penalty = min(self.failure_count * 0.1, 0.5)
        return max(0, 1.0 - load_penalty - memory_penalty - failure_penalty)


@dataclass
class DAGNode:
    node_id: int
    agent_id: int
    task_name: str
    dependencies: List[int] = field(default_factory=list)
    result: Optional[object] = None
    status: str = "pending"
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ExecutionDAG:
    dag_id: str
    nodes: List[DAGNode]
    edges: List[Tuple[int, int]]

    def get_ready_nodes(self, completed: Set[int]) -> List[DAGNode]:
        ready = []
        for node in self.nodes:
            if node.status == "pending":
                deps_met = all(d in completed for d in node.dependencies)
                if deps_met:
                    ready.append(node)
        return ready

    def is_complete(self) -> bool:
        return all(n.status in ("completed", "failed") for n in self.nodes)


class MetacognitiveObserver:
    """Observes system state and detects anomalies."""

    def __init__(self):
        self.state_history: List[Dict] = []
        self.anomaly_scores: List[float] = []

    def observe(self, agents: List[Agent], dag: ExecutionDAG) -> Dict:
        state = {
            "timestamp": time.time(),
            "agent_states": {a.agent_id: a.state.value for a in agents},
            "health_scores": {a.agent_id: a.health_score for a in agents},
            "dag_progress": sum(1 for n in dag.nodes if n.status == "completed") / max(len(dag.nodes), 1),
            "failed_agents": sum(1 for a in agents if a.state == AgentState.FAILED),
            "total_agents": len(agents)
        }
        self.state_history.append(state)
        return state

    def detect_anomaly(self, current_state: Dict) -> float:
        if len(self.state_history) < 2:
            return 0.0

        prev = self.state_history[-2]
        curr = current_state

        health_diff = sum(
            abs(curr["health_scores"].get(aid, 0) - prev["health_scores"].get(aid, 0))
            for aid in curr["health_scores"]
        )

        failure_diff = abs(curr["failed_agents"] - prev["failed_agents"])

        anomaly_score = health_diff * 0.5 + failure_diff * 0.3
        self.anomaly_scores.append(anomaly_score)
        return anomaly_score


class MetacognitiveEvaluator:
    """Evaluates system state and determines actions."""

    def __init__(self, anomaly_threshold: float = 0.3):
        self.anomaly_threshold = anomaly_threshold
        self.evaluation_history: List[Dict] = []

    def evaluate(self, observer: MetacognitiveObserver,
                 agents: List[Agent], dag: ExecutionDAG) -> Dict:
        current_state = observer.state_history[-1] if observer.state_history else {}
        anomaly_score = observer.detect_anomaly(current_state)

        failed_agents = [a for a in agents if a.state == AgentState.FAILED]
        overloaded_agents = [a for a in agents if a.task_load > 0.8]
        ready_nodes = dag.get_ready_nodes(
            {n.node_id for n in dag.nodes if n.status == "completed"}
        )

        needs_intervention = (
            anomaly_score > self.anomaly_threshold or
            len(failed_agents) > 0 or
            len(overloaded_agents) > len(agents) * 0.3
        )

        evaluation = {
            "anomaly_score": anomaly_score,
            "needs_intervention": needs_intervention,
            "failed_agents": [a.agent_id for a in failed_agents],
            "overloaded_agents": [a.agent_id for a in overloaded_agents],
            "ready_nodes": [n.node_id for n in ready_nodes],
            "dag_complete": dag.is_complete(),
            "recommended_actions": []
        }

        if needs_intervention:
            if failed_agents:
                evaluation["recommended_actions"].append("recover_failed")
            if overloaded_agents:
                evaluation["recommended_actions"].append("redistribute_load")
            if anomaly_score > self.anomaly_threshold * 2:
                evaluation["recommended_actions"].append("reconfigure_topology")

        self.evaluation_history.append(evaluation)
        return evaluation


class MetacognitiveModifier:
    """Modifies system behavior based on evaluations."""

    def __init__(self):
        self.modification_log: List[Dict] = []

    def apply_modification(self, agents: List[Agent],
                          dag: ExecutionDAG,
                          evaluation: Dict) -> List[Dict]:
        modifications = []

        if "recover_failed" in evaluation.get("recommended_actions", []):
            for agent in agents:
                if agent.state == AgentState.FAILED:
                    agent.state = AgentState.RECOVERING
                    agent.failure_count += 1
                    modifications.append({
                        "type": "recovery",
                        "agent_id": agent.agent_id,
                        "timestamp": time.time()
                    })

        if "redistribute_load" in evaluation.get("recommended_actions", []):
            overloaded = [a for a in agents if a.task_load > 0.8]
            underloaded = [a for a in agents if a.task_load < 0.3]
            if overloaded and underloaded:
                for over_agent in overloaded[:len(underloaded)]:
                    under_agent = underloaded[0]
                    transfer = over_agent.task_load * 0.3
                    over_agent.task_load -= transfer
                    under_agent.task_load += transfer
                    modifications.append({
                        "type": "load_transfer",
                        "from": over_agent.agent_id,
                        "to": under_agent.agent_id,
                        "amount": transfer,
                        "timestamp": time.time()
                    })

        if "reconfigure_topology" in evaluation.get("recommended_actions", []):
            for agent in agents:
                if agent.state == AgentState.RECOVERING:
                    agent.state = AgentState.IDLE
                    agent.task_load = 0.0
                    modifications.append({
                        "type": "topology_reconfigure",
                        "agent_id": agent.agent_id,
                        "timestamp": time.time()
                    })

        self.modification_log.extend(modifications)
        return modifications


class CognitiveResilienceLayer:
    """Main CRL metacognitive control plane."""

    def __init__(self):
        self.observer = MetacognitiveObserver()
        self.evaluator = MetacognitiveEvaluator()
        self.modifier = MetacognitiveModifier()
        self.cycle_count = 0

    def run_cycle(self, agents: List[Agent],
                  dag: ExecutionDAG) -> Dict:
        self.cycle_count += 1

        state = self.observer.observe(agents, dag)
        evaluation = self.evaluator.evaluate(self.observer, agents, dag)
        modifications = self.modifier.apply_modification(agents, dag, evaluation)

        return {
            "cycle": self.cycle_count,
            "state": state,
            "evaluation": evaluation,
            "modifications": modifications
        }

    def run_simulation(self, agents: List[Agent],
                       dag: ExecutionDAG,
                       num_cycles: int = 100,
                       failure_probability: float = 0.05) -> List[Dict]:
        results = []
        for cycle in range(num_cycles):
            if dag.is_complete():
                break

            if random.random() < failure_probability:
                active_agents = [a for a in agents if a.state != AgentState.FAILED]
                if active_agents:
                    random.choice(active_agents).state = AgentState.FAILED

            ready_nodes = dag.get_ready_nodes(
                {n.node_id for n in dag.nodes if n.status == "completed"}
            )

            for node in ready_nodes:
                agent = next((a for a in agents if a.agent_id == node.agent_id), None)
                if agent and agent.state in (AgentState.IDLE, AgentState.EXECUTING):
                    node.status = "completed"
                    node.end_time = time.time()
                    agent.task_load = min(1.0, agent.task_load + 0.1)

            result = self.run_cycle(agents, dag)
            results.append(result)

        return results


class SimulationRunner:
    """Main simulation runner for CRL paper."""

    def __init__(self, num_agents: int = 10, num_tasks: int = 20):
        self.num_agents = num_agents
        self.num_tasks = num_tasks

    def create_agents(self) -> List[Agent]:
        agents = []
        capabilities_pool = ["compute", "storage", "network", "ai", "sensor"]
        for i in range(self.num_agents):
            caps = random.sample(capabilities_pool, k=random.randint(2, 4))
            agents.append(Agent(
                agent_id=i,
                capabilities=caps,
                task_load=random.uniform(0.1, 0.5),
                memory_usage=random.uniform(0.2, 0.6)
            ))
        return agents

    def create_dag(self, agents: List[Agent]) -> ExecutionDAG:
        nodes = []
        edges = []
        for i in range(self.num_tasks):
            agent_id = random.choice([a.agent_id for a in agents])
            deps = []
            if i > 0 and random.random() < 0.4:
                deps.append(random.randint(0, i - 1))
            nodes.append(DAGNode(
                node_id=i,
                agent_id=agent_id,
                task_name=f"task_{i}",
                dependencies=deps
            ))

        for node in nodes:
            for dep in node.dependencies:
                edges.append((dep, node.node_id))

        return ExecutionDAG(
            dag_id="main_dag",
            nodes=nodes,
            edges=edges
        )

    def run_experiment(self) -> Dict:
        print("Creating agents and DAG...")
        agents = self.create_agents()
        dag = self.create_dag(agents)

        print(f"Running CRL simulation with {len(agents)} agents, {len(dag.nodes)} tasks...")
        crl = CognitiveResilienceLayer()
        results = crl.run_simulation(agents, dag, num_cycles=200)

        completed = sum(1 for n in dag.nodes if n.status == "completed")
        total_mods = sum(len(r["modifications"]) for r in results)
        avg_anomaly = np.mean([r["evaluation"]["anomaly_score"] for r in results]) if results else 0

        return {
            "total_tasks": len(dag.nodes),
            "completed_tasks": completed,
            "completion_rate": completed / max(len(dag.nodes), 1),
            "total_modifications": total_mods,
            "avg_anomaly_score": avg_anomaly,
            "cycles_run": len(results),
            "dag_complete": dag.is_complete()
        }

    def run_comparison(self) -> Dict:
        results_with_crl = []
        results_without_crl = []

        for trial in range(10):
            agents = self.create_agents()
            dag = self.create_dag(agents)

            crl = CognitiveResilienceLayer()
            trial_results = crl.run_simulation(
                agents, dag, num_cycles=200, failure_probability=0.20
            )
            completed = sum(1 for n in dag.nodes if n.status == "completed")
            results_with_crl.append(completed / max(len(dag.nodes), 1))

        for trial in range(10):
            agents = self.create_agents()
            dag = self.create_dag(agents)

            for cycle in range(200):
                if dag.is_complete():
                    break
                if random.random() < 0.20:
                    active = [a for a in agents if a.state != AgentState.FAILED]
                    if active:
                        random.choice(active).state = AgentState.FAILED
                ready_nodes = dag.get_ready_nodes(
                    {n.node_id for n in dag.nodes if n.status == "completed"}
                )
                for node in ready_nodes:
                    node.status = "completed"

            completed = sum(1 for n in dag.nodes if n.status == "completed")
            results_without_crl.append(completed / max(len(dag.nodes), 1))

        return {
            "with_crl": {
                "mean": np.mean(results_with_crl),
                "std": np.std(results_with_crl),
                "completion_rates": results_with_crl
            },
            "without_crl": {
                "mean": np.mean(results_without_crl),
                "std": np.std(results_without_crl),
                "completion_rates": results_without_crl
            }
        }


if __name__ == "__main__":
    np.random.seed(20260909)
    random.seed(20260909)
    print("=" * 60)
    print("Cognitive Resilience Layer as Metacognitive Control Plane")
    print("WSSE 2026 — Simulation Runner")
    print("=" * 60)

    runner = SimulationRunner(num_agents=10, num_tasks=20)

    print("\n--- Single Experiment ---")
    results = runner.run_experiment()
    print(f"Completion Rate: {results['completion_rate']:.2%}")
    print(f"Total Modifications: {results['total_modifications']}")
    print(f"Avg Anomaly Score: {results['avg_anomaly_score']:.4f}")
    print(f"DAG Complete: {results['dag_complete']}")

    print("\n--- Comparison: With vs Without CRL ---")
    comparison = runner.run_comparison()
    print(f"With CRL:    {comparison['with_crl']['mean']:.2%} ± {comparison['with_crl']['std']:.2%}")
    print(f"Without CRL: {comparison['without_crl']['mean']:.2%} ± {comparison['without_crl']['std']:.2%}")
    improvement = (comparison['with_crl']['mean'] - comparison['without_crl']['mean']) / max(comparison['without_crl']['mean'], 0.01) * 100
    print(f"Improvement: {improvement:.1f}%")
