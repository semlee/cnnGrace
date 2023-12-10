#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Split a string into a vector of substrings based on a delimiter
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);
    std::string token;
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main() {
    // Read X_train
    std::ifstream X_train_file("X_train.csv");
    std::vector<std::vector<double>> X_train;
    std::string line;
    while (std::getline(X_train_file, line)) {
        std::vector<std::string> tokens = split(line, ',');
        std::vector<double> row;
        for (const auto& token : tokens) {
            row.push_back(std::stod(token));
        }
        X_train.push_back(row);
    }

    // Read Y_train
    // ... similar process as above ...

    // Read mean
    // ... similar process as above ...

    return 0;
}
