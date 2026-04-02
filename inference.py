from agent import IssueTriageAgent
from envs.GitHubIssueTriageManager.server.environment import GitHubIssueTriageEnvironment
from envs.GitHubIssueTriageManager.server.grader import grade_episode

agent = IssueTriageAgent()
env = GitHubIssueTriageEnvironment(data_dir="data", strict_mode=True, live_github=False)

obs = env.reset()
done = False
result_grade = 0
while result_grade < 0.99:
    obs = env.reset()
    done = False
    while not done:
        # print(f"Observation: {obs.model_dump()}")
        action = agent.next_action(obs.model_dump())
        result = env.step(action)
        print(f"Action: {action}")
        obs = result.observation
        # print(f"Result: {result}")
        done = result.done

    target_state = env.state()
    grade = grade_episode(target_state)
    print(f"Final Grade: {grade}")
    result_grade = grade