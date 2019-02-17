#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <sstream>
#include <typeinfo>
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
      for (const auto& c : topic_frequencies_) {
        std::cout << c.first << " " << c.second << std::endl;
      }

      std::vector<std::string> words = split(lines_[i]);

      std::string topic = words[0];

      auto found = topic_frequencies_.find(topic);
      if (found != topic_frequencies_.end()) {
        #pragma omp critical
        {
        ++topic_frequencies_[topic];
        }
      } else {
        #pragma omp critical
        {
        topic_frequencies_[topic] = 1;
        }
      }

      for (size_t j = 1; j < words.size(); ++j) {
        for (const auto& suffix : suffixes) {
          #pragma omp critical
          {
          if (ends_with(words[j], suffix)) {
            words[j].substr(0, words[j].size() - suffix.size());
            }
          }
        }
      }

      // Tally frequencies
      count_word_frequencies(words);

      // Remake the words
      #pragma omp critical
      {
      lines_[i] = join(words, " ");
      }

      auto cfound = classified_text_.find(topic);
      if (cfound != classified_text_.end()) {
        #pragma omp critical
        {
        classified_text_[topic] += lines_[i];
        classified_text_[topic] += " ";
        }
      } else {
        #pragma omp critical
        {
        classified_text_[topic] = lines_[i];
        }
      }
    }

    for (const auto& topic : topics) {
      #pragma omp critical
      {
      auto found = word_frequencies_.find(topic);
      if (found != word_frequencies_.end()) {
        word_frequencies_.erase(topic);
      }
      }
    }
    }
  }

  void document::count_word_frequencies(const std::vector<std::string>& words) {
    for (const auto& word : words) {
      if (word_frequencies_.find(word) != word_frequencies_.end()) {
        ++word_frequencies_[word];
      } else {
        word_frequencies_[word] = 1;
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

  frequency_map Bayes::word_count(const std::vector<std::string>& words) {
    frequency_map counts;

    for (const auto& word : words) {
      ++counts[word];
    }

    return counts;
  }

  // sum the documents
  void Bayes::fit() {
    std::vector<std::string> vocabulary = doc_->extract_keys<std::string, int>(doc_->word_frequencies_);

    // Sum of document class counts;
    const int doc_sum = std::accumulate(
        doc_->topic_frequencies_.begin(), doc_->topic_frequencies_.end(), 0,
        [](const std::size_t prev, const auto& el) {
      return prev + el.second;
    });

    for (const auto& topic : doc_->topics) {
      probability_map estimates;
      const double class_proba = (double)doc_->topic_frequencies_[topic] / (double)doc_sum;
      std::cout << "Probability of topic " << topic << ": " << class_proba << std::endl;
      const std::string text = doc_->classified_text_[topic];
      const std::vector<std::string> topic_words = doc_->split(text);
      frequency_map word_count_in_topic = word_count(topic_words);
      const double n = (double)doc_->count_words_in_line(text);
      class_probabilities_[topic] = class_proba;

      for (const auto& word : vocabulary) {
        const double nk = (double)word_count_in_topic[word];
        const double estimate = std::log(nk + 1.0) / std::log(n + (double)vocabulary.size());
        estimates_[word] = estimate;
      }

      topic_word_probabilities_[topic] = estimates;
    }

    for (const auto& e : estimates_) {
      std::cout << e.first << " " << e.second << std::endl;
    }

    for (const auto& p : class_probabilities_) {
      std::cout << p.first << " " << p.second << std::endl;
    }
  }

  void Bayes::evaluate() {

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

  bae.fit();

  return EXIT_SUCCESS;
}
