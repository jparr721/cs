#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

template <class KTy, class Ty>
void PrintMap(std::map<KTy, Ty> map) {
    std::ofstream outFile;
    outFile.open("output.txt");
    for (auto it = map.cbegin(); it != map.cend(); ++it)
	outFile << it->first << ": " << it->second << "\n";
}

int main(void) {
    static const char* fileName= "input.txt";
    std::map<std::string, unsigned int> wordCount;
    std::ifstream inFile;
    inFile.open(fileName);
    char chars[] = ".";
    char nums[] = "2345";

    // Should be open, but ya know...
    if (inFile.is_open()) {
        while (inFile.good()) {
            std::string word;
            inFile >> word;

            for (unsigned int i = 0; i < word.length(); ++i) {
                if (isupper(word.at(i))) {
                    word.at(i) = tolower(word.at(i));
                }
                word.erase(std::remove(word.begin(), word.end(), chars[i]), word.end());
            }

            // check if word is already there
            if (wordCount.find(word) == wordCount.end()) {
                wordCount[word] = 1;
            } else {
                wordCount[word]++;
            }
        }
    }
    else {
        std::cerr << "Couldn't open file" << std::endl;
    }
    for (auto it = wordCount.begin(); it != wordCount.end();) {
        if (it->first == "23")
            it = wordCount.erase(it);
        if (it->first == "45")
            it = wordCount.erase(it);
        if (it->first == "the")
            it = wordCount.erase(it);
        if (it->first == "a")
            it = wordCount.erase(it);
        it++;
    }

    for (auto it = wordCount.begin(); it != wordCount.end();) {
        if (it->first == "!")
            it = wordCount.erase(it);

        if (it->first == "?")
            it = wordCount.erase(it);
        it++;
    }
    auto at = wordCount.find("@");
    wordCount.erase(at);
    PrintMap(wordCount);

    return 0;
}
