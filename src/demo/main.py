from src.demo.agent import Agent
from src.logger import setup_logging


def main():
    setup_logging()
    agent = Agent()
    task = agent.run("I want to buy a new phone")
    print(task.result or "")


if __name__ == "__main__":
    main()
