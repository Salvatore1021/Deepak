#include <iostream>
using namespace std;

class Calculator {
public:
    double add(double a, double b) {
        return a + b;
    }

    double subtract(double a, double b) {
        return a - b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    double divide(double a, double b) {
        if (b == 0) {
            cout << "Error: Cannot divide by zero!" << endl;
            return 0;
        }
        return a / b;
    }

    void displayCalculator() {
        cout << "=============================\n";
        cout << "         CALCULATOR          \n";
        cout << "=============================\n";
        cout << "|  7  |  8  |  9  |   /     |\n";
        cout << "|  4  |  5  |  6  |   *     |\n";
        cout << "|  1  |  2  |  3  |   -     |\n";
        cout << "|  0  |  .  |  =  |   +     |\n";
        cout << "=============================\n\n";
    }
};

int main() {
    Calculator calc;
    double num1, num2, result;
    char op;
    char choice = 'y';

    while (choice == 'y' || choice == 'Y') {
        calc.displayCalculator();

        cout << "Enter first number: ";
        cin >> num1;

        cout << "Enter operator (+, -, *, /): ";
        cin >> op;

        cout << "Enter second number: ";
        cin >> num2;

        switch (op) {
        case '+':
            result = calc.add(num1, num2);
            break;
        case '-':
            result = calc.subtract(num1, num2);
            break;
        case '*':
            result = calc.multiply(num1, num2);
            break;
        case '/':
            result = calc.divide(num1, num2);
            break;
        default:
            cout << "Invalid operator!" << endl;
            continue;
        }

        cout << "\nResult = " << result << endl;

        cout << "\nDo you want to calculate again? (y/n): ";
        cin >> choice;

        cout << "\n----------------------------------\n\n";
    }

    cout << "Thank you for using the calculator!" << endl;

    return 0;
}
