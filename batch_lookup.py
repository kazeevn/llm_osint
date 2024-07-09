import argparse
import re
from pathlib import Path
from multiprocessing import Pool
from omegaconf import OmegaConf
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from person_lookup import fetch_internet_content, ASK_PROMPT


def look_person_up(
    name: str,
    question: str,
    output_folder: Path,
    config: OmegaConf) -> None:
    """
    Look up a person and save the results to a file.
    Args:
        name: The name of the person
        question: The question to ask about the person
        output_folder: The folder to save the results to
        config: The LLM OSINT configuration
    """
    try:
        content = fetch_internet_content(name, config)
        file_name = re.sub(r"[^\w]", "", name).lower() + ".txt"
        with open(Path("internet_content", file_name), "wt", encoding="utf-8") as f:
            f.write(content)

        model = ChatOpenAI(**config.llm)
        result = model.invoke([HumanMessage(
                ASK_PROMPT.format(
                    name=name, internet_content=content, question=question))]).content
        with open(output_folder / file_name, "wt", encoding="utf-8") as f:
            f.write(result)
        print(f"Finished looking up {name}. Results:\n{result}")
        print(f"Results saved to {output_folder / file_name}")
    except Exception as e:
        print(f"Error looking up {name}: {e}")


def main():
    parser = argparse.ArgumentParser("Look up multiple people")
    parser.add_argument("names_file", type=Path,
        help="A file with names of people to look up, one per line")
    parser.add_argument("--ask", type=str, required=True,
        help="The question to ask about each person")
    parser.add_argument("--output-folder", type=Path, default="batch_results",
        help="The folder to save the results to")
    parser.add_argument("--n-jobs", type=int, default=2,
        help="The number of jobs to run in parallel")
    args = parser.parse_args()
    args.output_folder.mkdir(exist_ok=True)
    with open(args.names_file, "rt", encoding="utf-8") as f:
        names = f.read().splitlines()
    config = OmegaConf.load(Path(__file__).parent / "config.yaml")
    print(f"Looking up {len(names)} people...")
    with Pool(args.n_jobs) as pool:
        pool.starmap(
            look_person_up, [(name, args.ask, args.output_folder, config) for name in names])


if __name__ == "__main__":
    main()
