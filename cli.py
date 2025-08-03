from prediction import Predictor


class Cli:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def run(self) -> None:
        """Interactively classify user-inputted messages as spam or not."""
        print()
        try:
            while True:
                message = input("Enter message to classify (or 'quit'): ")
                if message.lower() == "quit":
                    break

                result = self.predictor.predict(message)
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print()

        except KeyboardInterrupt:
            print("\nProgram terminated by user")


if __name__ == "__main__":
    predictor = Predictor()
    cli = Cli(predictor)
    cli.run()
