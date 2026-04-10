from prediction import Predictor


def run_cli(predictor: Predictor):
    print()
    try:
        while True:
            message = input("Enter message to classify (or 'quit'): ")
            if message.lower() == "quit":
                break

            result = predictor.predict(message)
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print()

    except KeyboardInterrupt:
        print("\nProgram terminated by user")


if __name__ == "__main__":
    predictor = Predictor()
    run_cli(predictor)
