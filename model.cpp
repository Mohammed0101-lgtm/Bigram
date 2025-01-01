#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <random>

const int START = static_cast<int>('.');
const int END = static_cast<int>('.');

struct Vectorhash
{
    size_t operator()(const std::vector<int> &vec) const
    {
        size_t hash = 0;
        std::hash<int> hasher;
        for (int val : vec)
            hash ^= hasher(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

        return hash;
    }
};

struct Vectorequal
{
    bool operator()(const std::vector<int> &a, const std::vector<int> &b) const
    {
        return a == b;
    }
};

std::vector<std::string> read_file(const std::string &_filename)
{
    if (_filename.empty())
        throw std::invalid_argument("Filename cannot be empty");

    std::ifstream file(_filename, std::ios::in);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file : " + _filename);

    std::vector<std::string> ret;
    std::string line;

    while (std::getline(file, line))
        ret.push_back(line);

    file.close();
    return ret;
}

std::vector<std::vector<int>> encode(const std::vector<std::string> &_input)
{
    if (_input.empty())
        throw std::runtime_error("Input cannot be empty!");

    std::vector<std::vector<int>> ret;

    for (const std::string &w : _input)
    {
        ret.push_back({START, static_cast<int>(w[0])});
        int i = 0;
        for (; i < w.length() - 1; i++)
            ret.push_back({static_cast<int>(w[i]), static_cast<int>(w[i + 1])});
        
        ret.push_back({w[i + 1], END});
    }

    return ret;
}

std::unordered_map<std::vector<int>, int, Vectorhash, Vectorequal> embedding(const std::vector<std::vector<int>> &_input)
{
    if (_input.empty())
        throw std::invalid_argument("Cannot embed empty input!");

    std::unordered_map<std::vector<int>, int, Vectorhash, Vectorequal> ret;
    for (const std::vector<int> &v : _input)
        ret[v]++;

    return ret;
}

std::unordered_map<std::vector<int>, float, Vectorhash, Vectorequal>
probs(const std::unordered_map<std::vector<int>, int, Vectorhash, Vectorequal> &_map)
{
    float total = 0;
    for (const auto &pair : _map)
        total += pair.second;

    std::unordered_map<std::vector<int>, float, Vectorhash, Vectorequal> ret;
    if (total > 0)
        for (const auto &pair : _map)
            ret[pair.first] = static_cast<float>(pair.second) / total;

    return ret;
}

char sample_next_char(int current, const std::unordered_map<std::vector<int>, float, Vectorhash, Vectorequal> &probs)
{
    std::vector<std::pair<char, float>> candidates;
    float cumulative = 0;

    for (const auto &pair : probs)
    {
        if (pair.first[0] == current) 
        {
            candidates.emplace_back(static_cast<char>(pair.first[1]), pair.second);
            cumulative += pair.second;
        }
    }

    if (candidates.empty())
        return '\0';

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    float r = dis(gen);
    float running_sum = 0;

    for (const auto &candidate : candidates)
    {
        running_sum += candidate.second / cumulative;
        if (r <= running_sum)
            return candidate.first;
    }

    return candidates.back().first; 
}

std::string sample_sequence(const std::unordered_map<std::vector<int>, float, Vectorhash, Vectorequal> &probs, int max_length = 10)
{
    std::string result;
    char current = static_cast<char>(START);

    for (int i = 0; i < max_length; i++)
    {
        char next = sample_next_char(current, probs);
        if (next == '\0')
            break;

        result += next;
        current = next;
    }

    return result;
}

int main()
{
    auto input = read_file("input.txt");
    auto encoding = encode(input);
    auto counts = embedding(encoding);
    auto prob = probs(counts);

    for (const auto &pair : prob)
    {
        std::string w;
        for (const int &i : pair.first)
            w += static_cast<char>(i);

        std::cout << w << " : " << pair.second * 100 << '%' << std::endl;
    }

    std::cout << "\nGenerated Sequences:\n";
    for (int i = 0; i < 5; i++)
    {
        std::string sequence = sample_sequence(prob);
        std::cout << "Sample " << i + 1 << ": " << sequence << std::endl;
    }

    return 0;
}
