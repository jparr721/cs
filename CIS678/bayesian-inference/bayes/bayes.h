#ifndef BAYES_H_
#define BAYES_H_

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bayes {
  using frequency_map =
    std::unordered_map<std::string, int>;

  struct document {
    document(const std::string& doc_path);

    void load_document(const std::string& doc_path);
    void stem_document();
    void ostem_document();
    void count_word_frequencies(const std::vector<std::string>& words);

    std::size_t count_words_in_line(const std::string& line);

    bool ends_with(const std::string& word, const std::string& suffix);

    std::string join(const std::vector<std::string>& words, const std::string& delimiter) const;

    std::vector<std::string> split(std::string line);

    template<typename TK, typename TV>
    std::vector<TK> extract_keys(std::unordered_map<TK, TV> const& input_map) {
      std::vector<TK> retval;
      for (auto const& element : input_map) {
        retval.push_back(element.first);
      }
      return retval;
    }

    template<typename TK, typename TV>
    std::vector<TV> extract_values(std::unordered_map<TK, TV> const& input_map) {
      std::vector<TV> retval;
      for (auto const& element : input_map) {
        retval.push_back(element.second);
      }
      return retval;
    }

    std::vector<std::string> lines_;
    frequency_map topic_frequencies_;
    frequency_map word_frequencies_;
    std::unordered_map<std::string, std::string> classified_text_;

    const std::array<std::string, 3> suffixes{{"ed", "ing", "'s"}};
    const std::array<std::string, 20> topics{
      {
        "atheism", "graphics", "mswindows", "pc", "mac", "xwindows",
        "forsale", "autos", "motorcycles", "baseball", "hockey",
        "cryptology", "electronics", "medicine", "space", "christianity"
        "guns", "mideastpolitics", "politics", "religion"
      }
    };
  };

  class Bayes {
    public:
      Bayes(const document& d) : doc_(d) {};

      void fit();
      double predict();
    private:
      void classifier();

      std::optional<document> doc_;

      frequency_map estimates_;
      frequency_map class_probabilities_;
  };
} // namespace bayes

#endif // BAYES_H_
