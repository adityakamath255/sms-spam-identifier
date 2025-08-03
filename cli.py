from prediction import Predictor


class Cli:
    def __init__(self, pred: Predictor):
        self.pred = pred

    def run(self) -> None:
        """Interactively classify user-inputted messages as spam or not."""
        print()
        try:
            while True:
                message = input("Enter message to classify (or 'quit'): ")
                if message.lower() == "quit":
                    break

                prediction, probability = self.pred.predict(message)
                confidence = (
                    probability
                    if probability > 0.5
                    else 1 - probability
                )
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.2%}")
                print()

        except KeyboardInterrupt:
            print("\nProgram terminated by user")


if __name__ == "__main__":
    pred = Predictor()
    cli = Cli(pred)
    cli.run()
