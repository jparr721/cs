#ifndef BAYES_H_
#define BAYES_H_

#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bayes {
  using frequency_map =
    std::unordered_map<std::string_view, int>;

  struct document {
    document(const std::string& doc_path);

    void load_document(const std::string& doc_path);
    void stem_document();
    void count_word_frequencies(const std::vector<std::string>& words);

    int get_word_count(const std::string& word) const;

    bool ends_with(const std::string& word, const std::string& suffix);

    std::string join(const std::vector<std::string>& words, const std::string& delimiter) const;

    std::vector<std::string> split(std::string line);

    const frequency_map get_topic_frequencies() const;
    const frequency_map get_word_frequencies() const;

    std::vector<std::string> lines_;
    frequency_map topic_frequencies_;
    frequency_map word_frequencies_;
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
      Bayes(frequency_map topic_frequencies, frequency_map word_frequencies);

      double fit();
      double predict();
    private:
      void classifier();
  };
} // namespace bayes

#endif // BAYES_H_
