
from datasets import load_dataset


def main():
    dataset = load_dataset("truthful_qa", "generation")

    print("Dataset structure:")
    print(dataset)

    train = dataset["validation"]
    print(f"\nNumber of questions: {len(train)}")

    print("\nSample questions:")
    for example in train.select(range(3)):
        print(f"\nQuestion: {example['question']}")
        print(f"Best answer: {example['best_answer']}")
        print(f"Correct answers: {example['correct_answers']}")
        print(f"Incorrect answers: {example['incorrect_answers']}")


if __name__ == "__main__":
    main()
