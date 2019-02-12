#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <sstream>
#include "bayes.h"

namespace bayes {
  document::document(const std::string& doc_path) {
    load_document(doc_path);
    stem_document();
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

        #pragma omp atomic read
        if (topic_frequencies_.find(topic) != topic_frequencies_.end()) {
          #pragma omp atomic write
          ++topic_frequencies_[topic];
        } else {
          #pragma omp atomic write
          topic_frequencies_[topic] = 0;
        }

        for (size_t j = 0; j < words.size(); ++j) {
          for (const auto& suffix : suffixes) {
            if (ends_with(words[j], suffix)) {
              words[j].substr(0, words[j].size() - suffix.size());
            }
          }
        }

        // Tally frequencies
        count_word_frequencies(words);

        // Remake the words
        lines_[i] = join(words, " ");

        #pragma omp atomic read
        if (classified_text_.find(topic) != classified_text_.end()) {
          #pragma omp atmoic write
          classified_text_[topic] += lines_[i];
          classified_text_[topic] += " ";
        } else {
          #pragma omp atmoic write
          classified_text_[topic] = lines_[i];
        }

      }
    }

    // Clean up word frequencies outside of parallel region
    for (auto it = topics.begin(); it != topics.end(); ++it) {
      if (word_frequencies_[it]) {
        word_frequencies_.erase(it);
      }
    }
  }

  void document::count_word_frequencies(const std::vector<std::string>& words) {
    for (const auto& word : words) {
      if (word_frequencies_.find(word) != word_frequencies_.end()) {
        ++word_frequencies_[word];
      } else {
        word_frequencies_[word] = 0;
      }
    }
  }

  std::size_t document::count_words_in_line(const std::string& line) {
    const std::vector<std::string> woah_big_vector_here = split(line);

    return woah_big_vector_here.size();
  }

  bool document::ends_with(const std::string& word, const std::string& suffix) {
    return word.size() > suffix.size() &&
      word.substr(word.size() - suffix.size()) == suffix;
  }

  std::string document::join(const std::vector<std::string>& words, const std::string& delimiter) const {
    std::ostringstream oss;
    std::copy(words.begin(), words.end() - 1, std::ostream_iterator<std::string>(oss, delimiter.c_str()));
    oss << words.back();

    return oss.str();
  }

  std::vector<std::string> document::split(std::string line) {
    std::vector<std::string> result;
    std::istringstream iss(line);
    for (std::string line; iss >> line;)
      result.push_back(line);

    return result;
  }

  // sum the documents
  double Bayes::fit() {
    std::vector<double> probabilities;

    // Sum of document class counts;
    const int doc_sum = std::accumulate(topic_frequencies.begin(), topic_frequencies.end(), 0,
        [](const std::size_t prev, const auto& el) {
      return prev + el.second;
    });

    for (const auto& topic : topics_) {
      const double class_proba = topic_frequencies_[topic] / doc_sum;
      const std::string text = current_document_[topic];
      const int n = current_document_.count_words_in_line(text);
    }

    return 0.0;
  }

} // namespace bayes

int main(int argc, char** argv) {
  if (!argv[1]) {
    std::cout << "usage: bae data_path" << std::endl;
    return EXIT_SUCCESS;
  }

  const std::string path = argv[1];

  bayes::document d(path);
  bayes::Bayes bae(d);

  bae.predict();

  return EXIT_SUCCESS;
}
