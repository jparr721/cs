#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <sstream>
#include "bayes.h"

namespace bayes {
  document::document(const std::string& doc_path) {
    load_document(doc_path);
  }

  void document::load_document(const std::string& doc_path) {
    std::ifstream input(doc_path);
    if (!input.good()) throw std::invalid_argument("Invalid path specified");

    std::string line;
    while (std::getline(input, line)) {
      std::istringstream iss(line);
      lines_.push_back(std::string(iss.str()));
    }
  }

  void document::stem_document() {
    #pragma omp parallel
    {
      for (size_t i = 0; i < lines_.size(); ++i) {
        std::vector<std::string> words = split(lines_[i]);
        std::string_view topic = words[0];
        if (topic_frequencies_.find(topic) != topic_frequencies_.end()) {
          ++topic_frequencies_[topic];
        } else {
          topic_frequencies_[topic] = 0;
        }
        for (size_t i = 0; i < words.size(); ++i) {
          for (const auto& suffix : suffixes) {
            if (ends_with(words[i], suffix)) {
              words[i].substr(0, words[i].size() - suffix.size());
            }
          }
        }
      }
    }
  }

  void document::lemmatize_document() {

  }

  bool document::ends_with(const std::string& word, const std::string& suffix) {
    return word.size() > suffix.size() &&
      word.substr(word.size() - suffix.size()) == suffix;
  }

  std::vector<std::string> document::split(std::string line) {
    std::vector<std::string> result;
    std::istringstream iss(line);
    for (std::string line; iss >> line;)
      result.push_back(line);

    return result;
  }

} // namespace bayes

int main() {
  std::cout << "Sup" << std::endl;
  return EXIT_SUCCESS;
}
