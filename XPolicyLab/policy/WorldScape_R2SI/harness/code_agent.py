"""Run the shared code-writing Agent behind the frozen episode protocol."""
from .backends import Backend


class CodeAgentBackend(Backend):
    def __init__(self, spec, deployment):
        from ..runtime import AgentRuntime
        self.runtime = AgentRuntime(spec['runtime_config'],
                                    wam_catalog=spec['wam_skills'],
                                    deployment=deployment)

    def reset(self):
        self.runtime.reset()

    def observe(self, observation, executed_action):
        self.runtime.update_obs(observation)
        return False

    def plan(self):
        return self.runtime.get_action()

    def drain_trace(self):
        return self.runtime.drain_trace()

    def close(self):
        self.runtime.close()
