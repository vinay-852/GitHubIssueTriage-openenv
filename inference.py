from agent import IssueTriageAgent
from envs.GitHubIssueTriageManager.server.environment import GitHubIssueTriageEnvironment

agent = IssueTriageAgent()
env = GitHubIssueTriageEnvironment(data_dir="data", strict_mode=True, live_github=False)

obs = env.reset()
done = False

while not done:
    action = agent.next_action(obs.model_dump())
    result = env.step(action)
    print(f"Action: {action}")
    obs = result.observation
    done = result.done