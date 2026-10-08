import random
class HelloAIWorld:
    def __init__(self):
        self.weight = random.uniform(0,5)
        self.learning_rate = 0.1
    def predict(self,x):
        return self.weight * x
    def train(self,training_data, epochs=100):
        for epoch in range(epochs):
            total_error = 0
            for x, y_actual in training_data:
                y_predicted = self.predict(x)

                error = y_actual - y_predicted
                total_error += abs(error)

                self.weight += self.learning_rate * error * x
            if (epoch + 1) % 20 == 0:
                avg_error = total_error / len(training_data)
                print(f"Epoch {epoch +1}/{epochs} - Average Error: {avg_error:.2f} - Weight: {self.weight:.2f}")
        print("Training complete. Final weight:", f"{self.weight:.2f}")
def main():
    training_data = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]
    print("Training data:", training_data)
    ai = HelloAIWorld()
    ai.train(training_data, epochs=100)
    print()

    print("Testing predictions:")
    test_inputs = [6, 7, 10, 15, 12]
    for x in test_inputs:
        prediction = ai.predict(x)
        actual = 2 * x
        print(f"Input: {x}, Predicted: {prediction:.2f}, Actual: {actual}")


if __name__ == "__main__":
    main()