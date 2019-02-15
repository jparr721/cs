#ifndef BAYES_H_
#define BAYES_H_

#include <array>
#include <optional>
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

    std::size_t count_words_in_line(const std::string& line);

    bool ends_with(const std::string& word, const std::string& suffix);

    std::string join(const std::vector<std::string>& words, const std::string& delimiter) const;

    std::vector<std::string> split(std::string line);

    std::vector<std::string> lines_;
    frequency_map topic_frequencies_;
    frequency_map word_frequencies_;
    std::unordered_map<std::string_view, std::string> classified_text_;

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

      double fit();
      double predict();
    private:
      void classifier();

      std::optional<document> doc_;
  };
} // namespace bayes

#endif // BAYES_H_
