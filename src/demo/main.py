from src.demo.agent import Agent
from src.logger import setup_logging


def main():
    setup_logging()
    agent = Agent()
    task = agent.run("I want to buy a new phone")
    print(task.result.output if task.result is not None else "")


if __name__ == "__main__":
    main()
